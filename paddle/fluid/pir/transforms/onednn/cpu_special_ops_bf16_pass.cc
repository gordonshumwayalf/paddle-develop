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

#include "paddle/fluid/pir/transforms/onednn/cpu_special_ops_bf16_pass.h"

#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

template <class IrType1, class IrType2>
static pir::Type create_type(pir::Type type,
                             pir::Type out_dtype,
                             pir::IrContext *ctx) {
  auto input_type = type.dyn_cast<IrType1>();
  return IrType2::get(ctx,
                      out_dtype,
                      input_type.dims(),
                      input_type.data_layout(),
                      input_type.lod(),
                      input_type.offset());
}

// For ops like conv and concat, their input is sometimes packed as VectorType,
// hence current quantization doesn't work. Here we deal with them specifically.
class ConcatBf16QuantizePattern
    : public pir::OpRewritePattern<paddle::onednn::dialect::ConcatOp> {
 public:
  using pir::OpRewritePattern<
      paddle::onednn::dialect::ConcatOp>::OpRewritePattern;
  bool MatchAndRewrite(
      paddle::onednn::dialect::ConcatOp op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    // The input should come from combine.
    pir::CombineOp pre_op =
        pir::GetDefiningOpForInput(op, 0)->dyn_cast<pir::CombineOp>();
    if (!pre_op) return false;
    if (!pre_op.out().HasOneUse()) return false;

    auto op_attributes = op->attributes();
    auto onednn_data_type = op_attributes.at("mkldnn_data_type")
                                .dyn_cast<pir::StrAttribute>()
                                .AsString();
    if (onednn_data_type == "bfloat16") return false;
    op_attributes["mkldnn_data_type"] = rewriter.str_attr("bfloat16");

    auto combine_inputs = pre_op.inputs();

    for (size_t idx = 0; idx < combine_inputs.size(); idx++) {
      auto type = pre_op->operand_type(idx);
      // Currently we only process case where elements are all DenseTensor(s)
      if (!type.isa<pir::DenseTensorType>()) return false;
      // All Tensors should be fp32
      auto dtype = pir::GetDataTypeFromValue(pre_op->operand_source(idx));
      if (!dtype.isa<pir::Float32Type>()) return false;
    }

    pir::IrContext *ctx = rewriter.ir_context();

    std::unordered_map<std::string, pir::Attribute> q_attributes;
    q_attributes["scale"] = rewriter.float_attr(1.0f);
    q_attributes["shift"] = rewriter.float_attr(0.0f);
    q_attributes["is_negative_input"] = rewriter.bool_attr(false);
    q_attributes["output_format"] = rewriter.str_attr("NCHW");
    q_attributes["bfloat16"] = rewriter.bool_attr(true);

    // Insert quantize before combine
    std::vector<pir::Value> new_combine_inputs(combine_inputs.size());
    for (size_t idx = 0; idx < combine_inputs.size(); idx++) {
      paddle::onednn::dialect::QuantizeOp quant_op =
          rewriter.Build<paddle::onednn::dialect::QuantizeOp>(
              combine_inputs[idx], q_attributes);
      auto type = quant_op->result_type(0);
      pir::Type new_type =
          create_type<pir::DenseTensorType, paddle::dialect::DenseTensorType>(
              type, pir::BFloat16Type::get(ctx), ctx);
      quant_op->result(0).set_type(new_type);
      new_combine_inputs[idx] = quant_op.output();
    }
    // Create new combine
    pir::CombineOp new_combine =
        rewriter.Build<pir::CombineOp>(new_combine_inputs);
    rewriter.ReplaceAllUsesWith(pre_op.out(), new_combine.out());
    rewriter.EraseOp(pre_op);

    // Create new concat
    auto concat_info =
        ctx->GetRegisteredOpInfo(paddle::onednn::dialect::ConcatOp::name());
    if (!concat_info) return false;

    std::vector<pir::Type> op_item_inner_output_types;
    auto type = op->result_type(0);
    pir::Type new_type =
        create_type<pir::DenseTensorType, paddle::dialect::DenseTensorType>(
            type, pir::BFloat16Type::get(ctx), ctx);
    op_item_inner_output_types.push_back(new_type);

    paddle::onednn::dialect::ConcatOp new_concat =
        rewriter
            .Build({new_combine.out(), op.axis()},
                   op_attributes,
                   op_item_inner_output_types,
                   concat_info)
            ->dyn_cast<paddle::onednn::dialect::ConcatOp>();

    // Insert dequant op under concat
    std::unordered_map<std::string, pir::Attribute> dq_attributes;
    dq_attributes["scale"] = rewriter.float_attr(1.0f);
    dq_attributes["shift"] = rewriter.float_attr(0.0f);
    paddle::onednn::dialect::DequantizeOp dequant_op =
        rewriter.Build<paddle::onednn::dialect::DequantizeOp>(new_concat.out(),
                                                              dq_attributes);

    rewriter.ReplaceAllUsesWith(op.out(), dequant_op.output());
    rewriter.EraseOp(op);
    return true;
  }
};

class CPUSpecialOpsBf16Pass : public pir::PatternRewritePass {
 public:
  CPUSpecialOpsBf16Pass()
      : pir::PatternRewritePass("cpu_special_ops_bf16_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    uint32_t benefit = 100;

    auto concat_bf16_quant_pattern =
        std::make_unique<ConcatBf16QuantizePattern>(
            context, benefit--, std::vector<std::string>{});
    ps.Add(std::move(concat_bf16_quant_pattern));

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateCPUSpecialOpsBf16Pass() {
  return std::make_unique<CPUSpecialOpsBf16Pass>();
}

}  // namespace pir

REGISTER_IR_PASS(cpu_special_ops_bf16_pass, CPUSpecialOpsBf16Pass);
