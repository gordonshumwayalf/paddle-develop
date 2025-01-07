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

#pragma once

#include <string>

#include <functional>
#include "paddle/cinn/ir/stmt_visitors.h"
#include "paddle/cinn/ir/utils/stmt_converter.h"
#include "paddle/cinn/pass/pass_adaptor.h"

namespace cinn {
namespace optim {
namespace detail {
namespace {
template <typename PassT, typename IRScopeRefT>
LogicalResult RunPasses(const std::vector<std::unique_ptr<PassT>>& passes,
                        IRScopeRefT scope) {
  for (auto& pass : passes) {
    if ((pass->Run(scope)).failed()) {
      VLOG(3) << "Failed to run pass: " << pass->name();
      return LogicalResult::failure();
    }
  }
  return LogicalResult::success();
}
}  // namespace

LogicalResult FuncPassAdaptor::Run(
    ir::LoweredFunc func,
    const std::vector<std::unique_ptr<FuncPass>>& passes) {
  return RunPasses(passes, func);
}

LogicalResult FuncPassAdaptor::Run(
    ir::stmt::BlockRef block,
    const std::vector<std::unique_ptr<FuncPass>>& passes) {
  // Would not adapt FuncPass on block scope.
  return LogicalResult::failure();
}

LogicalResult BlockPassAdaptor::Run(
    ir::LoweredFunc func,
    const std::vector<std::unique_ptr<BlockPass>>& passes) {
  return Run(func->body_block, passes);
}

LogicalResult BlockPassAdaptor::Run(
    ir::stmt::BlockRef block,
    const std::vector<std::unique_ptr<BlockPass>>& passes) {
  std::vector<ir::stmt::StmtRef> new_stmts = block->stmts();
  for (ir::stmt::StmtRef inner_stmt : new_stmts) {
    std::vector<ir::stmt::BlockRef> inner_blocks = inner_stmt->block_fields();
    for (ir::stmt::BlockRef inner_block : inner_blocks) {
      if (Run(inner_block, passes).failed()) return LogicalResult::failure();
    }
    inner_stmt->set_block_fields(inner_blocks);
  }
  block->set_stmts(new_stmts);
  return RunPasses(passes, block);
}

LogicalResult StmtPassAdaptor::Run(
    ir::LoweredFunc func,
    const std::vector<std::unique_ptr<StmtPass>>& passes) {
  return Run(func->body_block, passes);
}

LogicalResult StmtPassAdaptor::Run(
    ir::stmt::BlockRef block,
    const std::vector<std::unique_ptr<StmtPass>>& passes) {
  LogicalResult res = LogicalResult::success();
  ir::stmt::Mutate(
      block,
      [&](ir::stmt::StmtRef stmt) {},
      [&](ir::stmt::StmtRef stmt) {
        if (RunPasses(passes, stmt).failed()) {
          res = LogicalResult::failure();
        }
      });
  return res;
}

namespace {
using ExprMutateFuncT = std::function<LogicalResult(ir::Expr expr)>;
class StmtToExprPassAdaptor : public StmtPass {
 public:
  explicit StmtToExprPassAdaptor(const ExprMutateFuncT& func)
      : StmtPass("stmt to expr pass adaptor"), mutator_(func) {}
  virtual LogicalResult Run(ir::stmt::StmtRef stmt) {
    return mutator_.VisitStmt(stmt);
  }

 private:
  class LocalExprMutator : public ir::stmt::StmtMutator<LogicalResult> {
   public:
    explicit LocalExprMutator(const ExprMutateFuncT& expr_mutator)
        : expr_mutator_(expr_mutator) {}

    LogicalResult VisitStmt(ir::stmt::StmtRef stmt) override {
      return ir::stmt::StmtMutator<LogicalResult>::VisitStmt(stmt);
    }

   private:
    ExprMutateFuncT expr_mutator_;
#define __(stmt__) LogicalResult VisitStmt(ir::stmt::stmt__ stmt) override;
    NODETY_FORALL_STMT(__)
#undef __
  };
  LocalExprMutator mutator_;
};

#define MUTATE_EXPR(expr__) \
  if (expr_mutator_(expr__).failed()) return LogicalResult::failure();

LogicalResult StmtToExprPassAdaptor::LocalExprMutator::VisitStmt(
    ir::stmt::Let stmt) {
  ir::Expr symbol = stmt->symbol();
  ir::Expr body = stmt->body();
  MUTATE_EXPR(symbol);
  if (body.defined()) {
    MUTATE_EXPR(body);
  }
  stmt->set_symbol(symbol);
  stmt->set_body(body);
  return LogicalResult::success();
}

LogicalResult StmtToExprPassAdaptor::LocalExprMutator::VisitStmt(
    ir::stmt::Store stmt) {
  ir::Expr value = stmt->value();
  ir::Expr tensor = stmt->tensor();
  std::vector<ir::Expr> indices = stmt->indices();
  MUTATE_EXPR(value);
  MUTATE_EXPR(tensor);
  for (ir::Expr indice : indices) {
    MUTATE_EXPR(indice);
  }
  stmt->set_value(value);
  stmt->set_tensor(tensor);
  stmt->set_indices(indices);
  return LogicalResult::success();
}

LogicalResult StmtToExprPassAdaptor::LocalExprMutator::VisitStmt(
    ir::stmt::Alloc stmt) {
  std::vector<ir::Expr> extents = stmt->extents();
  ir::Expr condition = stmt->condition();
  ir::Expr body = stmt->body();
  for (ir::Expr extent : extents) {
    MUTATE_EXPR(extent);
  }
  if (condition.defined()) {
    MUTATE_EXPR(condition);
  }
  if (body.defined()) {
    MUTATE_EXPR(body);
  }
  stmt->set_extents(extents);
  stmt->set_condition(condition);
  stmt->set_body(body);
  return LogicalResult::success();
}

LogicalResult StmtToExprPassAdaptor::LocalExprMutator::VisitStmt(
    ir::stmt::Free stmt) {
  ir::Expr destination = stmt->destination();
  MUTATE_EXPR(destination);
  stmt->set_destination(destination);
  return LogicalResult::success();
}

LogicalResult StmtToExprPassAdaptor::LocalExprMutator::VisitStmt(
    ir::stmt::IfThenElse stmt) {
  ir::Expr condition = stmt->condition();
  MUTATE_EXPR(condition);
  stmt->set_condition(condition);
  return LogicalResult::success();
}

LogicalResult StmtToExprPassAdaptor::LocalExprMutator::VisitStmt(
    ir::stmt::For stmt) {
  ir::Expr min = stmt->min();
  ir::Expr extent = stmt->extent();
  MUTATE_EXPR(min);
  MUTATE_EXPR(extent);
  stmt->set_min(min);
  stmt->set_extent(extent);
  return LogicalResult::success();
}

LogicalResult StmtToExprPassAdaptor::LocalExprMutator::VisitStmt(
    ir::stmt::Schedule stmt) {
  std::vector<ir::Var> iter_vars = stmt->iter_vars();
  std::vector<ir::Expr> iter_values = stmt->iter_values();
  std::vector<ir::Expr> read_buffers = stmt->read_buffers();
  std::vector<ir::Expr> write_buffers = stmt->write_buffers();

  for (ir::Var iter_var : iter_vars) {
    if (iter_var->lower_bound.defined()) {
      MUTATE_EXPR(iter_var->lower_bound);
    }
    if (iter_var->upper_bound.defined()) {
      MUTATE_EXPR(iter_var->upper_bound);
    }
  }
  for (ir::Expr iter_value : iter_values) {
    MUTATE_EXPR(iter_value);
  }
  for (ir::Expr read_buffer : read_buffers) {
    MUTATE_EXPR(read_buffer);
  }
  for (ir::Expr write_buffer : write_buffers) {
    MUTATE_EXPR(write_buffer);
  }

  stmt->set_iter_vars(iter_vars);
  stmt->set_iter_values(iter_values);
  stmt->set_read_buffers(read_buffers);
  stmt->set_write_buffers(write_buffers);
  return LogicalResult::success();
}

LogicalResult StmtToExprPassAdaptor::LocalExprMutator::VisitStmt(
    ir::stmt::Evaluate stmt) {
  ir::Expr value = stmt->value();
  MUTATE_EXPR(value);
  stmt->set_value(value);
  return LogicalResult::success();
}
#undef MUTATE_EXPR
}  // namespace

LogicalResult ExprPassAdaptor::Run(
    ir::LoweredFunc func,
    const std::vector<std::unique_ptr<ExprPass>>& passes) {
  return Run(func->body_block, passes);
}

LogicalResult ExprPassAdaptor::Run(
    ir::stmt::BlockRef block,
    const std::vector<std::unique_ptr<ExprPass>>& passes) {
  std::vector<std::unique_ptr<StmtPass>> stmt_passes;
  stmt_passes.emplace_back(std::move(std::make_unique<StmtToExprPassAdaptor>(
      [&](ir::Expr expr) { return RunPasses(passes, expr); })));
  StmtPassAdaptor stmt_pass_adaptor;
  return stmt_pass_adaptor.Run(block, stmt_passes);
}

}  // namespace detail
}  // namespace optim
}  // namespace cinn
