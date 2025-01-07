// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/optim/rearrange_load_instruction.h"

#include <vector>
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"

PD_DECLARE_bool(cinn_enable_rearrange_load);

namespace cinn {
namespace optim {
namespace {

constexpr int MaxRearrangeLoadNum = 8;

template <typename NodeTy>
bool ContainsExprNode(const ir::Expr& expr) {
  auto res = ir::ir_utils::CollectIRNodes(
      expr,
      [](const ir::Expr* x) { return x->As<NodeTy>(); },
      /* uniq_target = */ true);
  return !res.empty();
}

/**
 * Calculate the buffer size as a constant. For dynamic dims, since they are
 * difficult to compare, we just estimate them to be 32.
 * Note: this is a heuristic optimization, so the exact number is not very
 * important.
 */
int64_t EstimateBufferSize(const ir::Buffer& buffer) {
  int64_t size = 1;
  for (auto& dim_size : buffer->shape) {
    if (dim_size.is_constant()) {
      size *= dim_size.as_int64();
    } else {
      size *= 32;
    }
  }
  return size;
}

std::vector<std::string> SortLoadsByBufferSizes(
    const std::unordered_map<std::string, const ir::Expr*>& load_map,
    std::vector<std::string> load_list) {
  // Calculate the buffer sizes of loads (with estimation).
  std::map<ir::Buffer, int64_t> buffer_size_map;
  for (auto& [_, load_expr] : load_map) {
    auto& buffer = load_expr->As<ir::Load>()->tensor.as_tensor()->buffer;
    if (buffer_size_map.count(buffer)) {
      continue;
    }
    buffer_size_map[buffer] = EstimateBufferSize(buffer);
  }

  const auto GetBufferSize = [&](const std::string& key) {
    auto& buffer = load_map.at(key)->As<ir::Load>()->tensor.as_tensor()->buffer;
    return buffer_size_map[buffer];
  };

  // Sort loads by their buffer sizes from large to small.
  // Note: we use stable sort here, because for equal-size loads, we want to
  // keep their original order.
  std::stable_sort(load_list.begin(),
                   load_list.end(),
                   [&](const std::string& key1, const std::string& key2) {
                     return GetBufferSize(key1) > GetBufferSize(key2);
                   });
  return load_list;
}

struct LoadCollector : public ir::IRMutator<> {
  explicit LoadCollector(const std::set<ir::Buffer>& locally_defined_buffers)
      : locally_defined_buffers_(locally_defined_buffers) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  // Collect loads that meet the following criteria:
  // 1) It is loading from global memory. Local loads are simply register reads
  //    and do not require rearrangement.
  // 2) The value being loaded is not defined locally by a previous store. In
  //    such cases, the value resides in a register rather than in memory, thus
  //    doesn't need rearrangement. This criteria also prevents data-dependency
  //    harzards.
  // 3) It doesn't contains indirect indices (i.e. loads within indices).
  //    Indirect indices are hard to manage and are seldom seem, so we choose
  //    not to handle them.
  void Visit(const ir::Load* op, ir::Expr* expr) override {
    auto& buffer = op->tensor.as_tensor()->buffer;
    if (buffer->memory_type != ir::MemoryType::Heap) {
      return;
    }
    if (locally_defined_buffers_.count(buffer) > 0) {
      return;
    }
    for (auto& index_expr : op->indices) {
      if (ContainsExprNode<ir::Load>(index_expr)) {
        return;
      }
    }
    std::string key = utils::GetStreamCnt(*expr);
    CollectLoad(key, expr);
  }

  // Handle Select as a special op.
  // Since Select evaluates only one of its two branches, we can rearrange a
  // load in Select only if the load appears in both branches, otherwise we may
  // violate the control dependency.
  void Visit(const ir::Select* op, ir::Expr* expr) override {
    auto* node = expr->As<ir::Select>();
    ir::IRMutator<>::Visit(&node->condition, &node->condition);

    LoadCollector true_collector(locally_defined_buffers_);
    true_collector(&node->true_value);
    LoadCollector false_collector(locally_defined_buffers_);
    false_collector(&node->false_value);

    for (auto& key : true_collector.load_list_) {
      if (false_collector.load_map_.count(key) > 0) {
        CollectLoad(key, true_collector.load_map_[key]);
      }
    }
  }

  void CollectLoad(const std::string& key, const ir::Expr* expr) {
    auto [_, is_first] = load_map_.emplace(key, expr);
    if (is_first) {
      load_list_.push_back(key);
    }
  }

 public:
  // map from the signatures of loads to the load nodes
  std::unordered_map<std::string, const ir::Expr*> load_map_;
  // list of the signatures of loads in the order they are visited
  std::vector<std::string> load_list_;

 private:
  const std::set<ir::Buffer>& locally_defined_buffers_;
};

struct LoadReplacer : public ir::IRMutator<> {
  explicit LoadReplacer(const std::unordered_map<std::string, ir::Var>& var_map)
      : var_map_(var_map) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Load* op, ir::Expr* expr) override {
    std::string key = utils::GetStreamCnt(*expr);
    if (var_map_.count(key) > 0) {
      *expr = Expr(var_map_.at(key));
    }
  }

  const std::unordered_map<std::string, ir::Var>& var_map_;
};

struct RearrangeLoadInstructionMutator : public ir::IRMutator<> {
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  // A block is a leaf block if it is inside at least one loop, and all of its
  // stmts are schedule blocks.
  bool IsLeafBlock(const ir::Block& block) {
    if (parent_loops_.empty()) {
      return false;
    }
    for (auto& stmt : block.stmts) {
      if (!stmt.As<ir::ScheduleBlockRealize>()) {
        return false;
      }
      auto* node = stmt.As<ir::ScheduleBlockRealize>()
                       ->schedule_block.As<ir::ScheduleBlock>();
      if (node->name.substr(0, 4) == "root") {
        return false;
      }
    }
    return true;
  }

  // Local buffer initialization is like:
  //    var_1_local[0] = var_1[blockIdx.x],
  // where the lhs is a local buffer and the rhs is a single load.
  bool IsLocalBufferInit(const ir::Store& store) {
    auto& store_buffer = store.tensor.as_tensor()->buffer;
    return store_buffer->memory_type == ir::MemoryType::GPULocal &&
           store.value.As<ir::Load>();
  }

  void DoRearrangeLoadInstruction(ir::Block* block) {
    // Step 1. Collect loads in each schedule block under this block.
    // Requirements:
    // 1) The schedule block cannot contain IfThenElse, or we will violate the
    //    control dependency. Schedule blocks that have IfThenElse usually don't
    //    benefit from rearranging loads, so it's ok to skip them.
    // 2) The schedule block is not local buffer initialization, because when
    //    initializing the local buffer with a load, we have already rearranged
    //    that load.
    // 3) There are more constrains on the loads to collect, see LoadCollector
    //    for details.
    LoadCollector collector(locally_defined_buffers_);
    for (auto& stmt : block->stmts) {
      ir::Expr store = ir::analyzer::GetStoreOfSBlock(stmt);
      auto* store_node = store.As<ir::Store>();
      if (ContainsExprNode<ir::IfThenElse>(stmt)) continue;
      if (IsLocalBufferInit(*store_node)) continue;
      collector(&store_node->value);
    }

    // Step 2. Sort the loads by their buffer sizes from large to small, and
    //    only keep the first `MaxRearrangeLoadNum` loads.
    // Performance concerns:
    // 1) Larger buffers need more time to access, so we should issue their
    //    corresponding loads earlier.
    // 2) Rearranged loads will consume registers, so we should set a limit
    //    to prevent register overflow.
    std::vector<std::string> load_list =
        SortLoadsByBufferSizes(collector.load_map_, collector.load_list_);
    if (load_list.size() > MaxRearrangeLoadNum) {
      load_list.resize(MaxRearrangeLoadNum);
    }

    // Step 3. Create loads with Let at the beginning of the block.
    std::vector<ir::Expr> new_stmts;
    std::unordered_map<std::string, ir::Var> var_map;
    for (auto& key : load_list) {
      auto* load_expr = collector.load_map_[key];
      auto* tensor = load_expr->As<ir::Load>()->tensor.as_tensor();
      ir::Var local_var = ir::Var(common::UniqName(tensor->name + "_local"),
                                  tensor->buffer->dtype);
      ir::Expr let_expr = ir::Let::Make(local_var, *load_expr);
      new_stmts.push_back(let_expr);
      var_map[key] = local_var;
    }

    // Step 4. Replace loads in schedule blocks with the above Let vars.
    LoadReplacer replacer(var_map);
    for (auto& stmt : block->stmts) {
      replacer(&stmt);
      new_stmts.push_back(stmt);
    }
    block->stmts = std::move(new_stmts);
  }

  void Visit(const ir::Block* op, ir::Expr* expr) override {
    auto* node = expr->As<ir::Block>();
    ir::IRMutator<>::Visit(op, expr);
    if (IsLeafBlock(*op)) {
      DoRearrangeLoadInstruction(node);
    }
  }

  void Visit(const ir::ScheduleBlockRealize* op, ir::Expr* expr) override {
    auto* block_node = op->schedule_block.As<ir::ScheduleBlock>();
    if (block_node->name.substr(0, 4) == "root") {
      ir::IRMutator<>::Visit(op, expr);
      return;
    }
    for (auto& buffer_range : block_node->write_buffers) {
      auto& write_buffer = buffer_range.As<ir::_BufferRange_>()->buffer;
      locally_defined_buffers_.insert(write_buffer.as_buffer_ref());
    }
  }

  void Visit(const ir::For* op, ir::Expr* expr) override {
    parent_loops_.push_back(op);
    ir::IRMutator<>::Visit(op, expr);
    parent_loops_.pop_back();
  }

 private:
  // Buffers whose values are defined locally inside this function.
  // Note: even if a buffer is allocated on global memory, its value may be
  // assigned locally. If so, it also belongs to this set.
  std::set<ir::Buffer> locally_defined_buffers_;

  std::vector<const ir::For*> parent_loops_;
};

}  // namespace

void RearrangeLoadInstruction(Expr* expr) {
  if (!FLAGS_cinn_enable_rearrange_load) return;
  RearrangeLoadInstructionMutator mutator;
  mutator(expr);
}

}  // namespace optim
}  // namespace cinn
