// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/group_schedule/config/group_tile_config.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_impl.h"

namespace cinn {
namespace ir {

using TileConfig = ScheduleConfig::TileConfig;
using TileConfigMap =
    std::unordered_map<BucketInfo, TileConfig, BucketInfoHash>;

namespace {

const int kMaxNumel = BucketInfo::kMaxNumel;

int64_t CeilPow2(int64_t n) {
  int64_t pow = 1;
  while (pow < n) {
    pow *= 2;
  }
  return pow;
}

int64_t FloorPow2(int64_t n) {
  int64_t pow = 1;
  while (pow * 2 <= n) {
    pow *= 2;
  }
  return pow;
}

int64_t CeilDiv(int64_t n, int64_t m) { return (n + m - 1) / m; }

int64_t Trim(int64_t n, int64_t min, int64_t max) {
  return std::min(std::max(n, min), max);
}

}  // namespace

BucketInfo::BucketInfo(int sp_lower_bound,
                       int sp_upper_bound,
                       int rb_lower_bound,
                       int rb_upper_bound,
                       bool sp_is_dynamic = false,
                       bool rb_is_dynamic = false) {
  if (sp_is_dynamic || sp_lower_bound != 1 || sp_upper_bound != 1) {
    BucketInfo::Dimension sp_dimension(
        sp_lower_bound, sp_upper_bound, "S", sp_is_dynamic);
    this->space.push_back(sp_dimension);
  }
  if (rb_is_dynamic || rb_lower_bound != 1 || rb_upper_bound != 1) {
    BucketInfo::Dimension rb_dimension(
        rb_lower_bound, rb_upper_bound, "R", rb_is_dynamic);
    this->space.push_back(rb_dimension);
  }
  if (this->space.empty()) {
    this->space.emplace_back(1, 1, "S", /* is_dynamic = */ false);
  }
}

BucketInfo::BucketInfo(const std::vector<BucketInfo::Dimension>& dims) {
  for (auto& dim : dims) {
    if (dim.is_dynamic || dim.lower_bound != 1 || dim.upper_bound != 1) {
      this->space.push_back(dim);
    }
  }
  if (this->space.empty()) {
    this->space.emplace_back(1, 1, "S", /* is_dynamic = */ false);
  }
}

bool BucketInfo::operator==(const BucketInfo& other) const {
  if (this->bucket_priority != other.bucket_priority) {
    return false;
  }
  if (this->space.size() != other.space.size()) {
    return false;
  }
  int length = this->space.size();
  for (int i = 0; i < length; i++) {
    if (this->space[i].is_dynamic != other.space[i].is_dynamic ||
        this->space[i].iter_type != other.space[i].iter_type ||
        this->space[i].lower_bound != other.space[i].lower_bound ||
        this->space[i].upper_bound != other.space[i].upper_bound) {
      return false;
    }
  }
  return true;
}

std::string BucketInfo::ToString() const {
  std::stringstream ss;
  ss << "BucketInfo: [";
  for (const auto& dim : space) {
    ss << dim.iter_type << "(" << dim.lower_bound << " - " << dim.upper_bound
       << "), ";
  }
  ss << "]";
  return ss.str();
}

ir::IterSpaceType GetIterSpaceType(
    const std::shared_ptr<FusionGroupInfo>& group_info,
    const std::set<int64_t>& reduce_dim_loc) {
  // Sort loops by their loop strides (if loop strides are known), so that we
  // get the loop index corresponding to the original memory layout.
  std::vector<int> loop_index(group_info->loop_ranges.size());
  std::iota(loop_index.begin(), loop_index.end(), 0);
  if (!group_info->loop_strides.empty()) {
    std::stable_sort(loop_index.begin(), loop_index.end(), [&](int i, int j) {
      return group_info->loop_strides[i] > group_info->loop_strides[j];
    });
  }

  ir::IterSpaceType iter_space_type;
  for (int i : loop_index) {
    int64_t loop_range = group_info->loop_ranges[i];
    if (loop_range == 1) {
      continue;
    }
    std::string iter_type = reduce_dim_loc.count(i) > 0 ? "R" : "S";
    std::string shape_type = loop_range == -1 ? "dynamic" : "static";
    if (iter_space_type.empty() || iter_space_type.back().first != iter_type) {
      iter_space_type.push_back({iter_type, shape_type});
    } else if (shape_type == "dynamic") {
      iter_space_type.back().second = shape_type;
    }
  }

  if (iter_space_type.empty()) {
    iter_space_type.push_back({"S", "static"});
  }
  return iter_space_type;
}

std::shared_ptr<ScheduleConfig::BaseInfo> InitBasicInfo(
    const std::shared_ptr<FusionGroupInfo>& group_info) {
  std::shared_ptr<ScheduleConfig::BaseInfo> base_info =
      std::make_shared<ScheduleConfig::BaseInfo>();
  base_info->reduce_axis = group_info->reduce_axis;
  base_info->loop_ranges = group_info->loop_ranges;
  base_info->loop_strides = group_info->loop_strides;
  base_info->can_apply_grid_reduce = group_info->can_apply_grid_reduce;

  std::set<int64_t> reduce_dim_loc(group_info->reduce_axis.begin(),
                                   group_info->reduce_axis.end());

  base_info->spatial_numel = 1;
  base_info->reduce_numel = 1;
  for (int64_t i = 0; i < base_info->loop_ranges.size(); ++i) {
    if (reduce_dim_loc.count(i)) {
      if (group_info->loop_ranges[i] == -1)
        base_info->has_dynamic_reduce = true;
      base_info->reduce_numel *= group_info->loop_ranges[i];
    } else {
      if (group_info->loop_ranges[i] == -1)
        base_info->has_dynamic_spatial = true;
      base_info->spatial_numel *= group_info->loop_ranges[i];
    }
  }

  base_info->iter_space_type = GetIterSpaceType(group_info, reduce_dim_loc);
  return base_info;
}

TileConfigMap BuildPureStaticShapeConfig(
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info,
    const common::Target& target) {
  const auto& last_dim = base_info->iter_space_type.back().first;
  const int sm_count = target.get_multi_processor_count();
  int64_t spatial_numel = base_info->spatial_numel;
  int64_t reduce_numel = base_info->reduce_numel;
  ReduceMethod reduce_method = NoneReduceMethod();

  // 1. Allocate spatial/reduce threads
  // Principals:
  //   1) The low 32 threads are assigned to the last dimension to ensure
  //      coalesced memory access.
  //   2) The remaining threads are assigned to either the reduce or spatial
  //      dimension, based on which dimension is the bottleneck.
  int64_t sp_thread_num = 1;
  int64_t rd_thread_num = 1;
  if (last_dim == "R") {
    rd_thread_num = 32;
    int64_t remain_reduce_numel = CeilDiv(reduce_numel, 32);
    if ((remain_reduce_numel <= 8 && spatial_numel > 1) ||
        (spatial_numel > remain_reduce_numel * 128)) {
      sp_thread_num = Trim(spatial_numel, 1, 8);
      reduce_method = WarpReduceMethod();
    } else {
      rd_thread_num *= Trim(remain_reduce_numel, 1, 32);
      reduce_method = BlockReduceMethod();
    }
  } else {  // last_dim == "S"
    sp_thread_num = 32;
    int64_t remain_spatial_numel = CeilDiv(spatial_numel, 32);
    if (reduce_numel <= 16) {
      sp_thread_num *= Trim(remain_spatial_numel, 1, 8);
    } else {
      rd_thread_num = Trim(reduce_numel, 1, 16);
      reduce_method = DiscreteReduceMethod();
    }
  }
  spatial_numel = CeilDiv(spatial_numel, sp_thread_num);
  reduce_numel = CeilDiv(reduce_numel, rd_thread_num);

  // 2. Allocate grid reduce blocks
  // Principals:
  //   1) Choose the largest reduce block number as long as the total number of
  //      blocks (rd_block * sp_block) doesn't exceed the SM count.
  //   2) Do not allocate too many reduce blocks when reduce_numel is small.
  int64_t rd_block_num = [&]() -> int64_t {
    if (!base_info->can_apply_grid_reduce) {
      return 1;
    }
    int64_t expected = sm_count / spatial_numel;
    int64_t limit_when_small = CeilDiv(reduce_numel, 32);
    return FloorPow2(Trim(expected, 1, limit_when_small));
  }();

  // 3. Allocate spatial inner loops
  // Principals:
  //   1) Choose an appropriate spatial inner loop number so that the spatial
  //      block number had better not exceed 4 times of SM count.
  //   2) Loops can only be assigned to either reduce or spatial, otherwise the
  //      index expression will be complex.
  int64_t sp_inner_num = [&]() -> int64_t {
    int64_t rd_inner_num = CeilDiv(reduce_numel, rd_block_num);
    if (rd_inner_num > 1) {
      return 1;
    }
    int64_t expected = spatial_numel / (sm_count * 4);
    return CeilPow2(Trim(expected, 1, 4));
  }();

  int64_t sp_upper_bound = base_info->spatial_numel > 1 ? kMaxNumel : 1;
  int64_t rd_upper_bound = base_info->reduce_numel > 1 ? kMaxNumel : 1;
  int64_t warp_num = Trim(sp_thread_num * rd_thread_num / 32, 1, 32);
  BucketInfo bucket_info{1, sp_upper_bound, 1, rd_upper_bound};
  TileConfig tile_config{warp_num,
                         /* tree_reduce_num = */ rd_thread_num,
                         /* grid_reduce_num = */ rd_block_num,
                         /* spatial_inner_num = */ sp_inner_num,
                         reduce_method};
  return {{bucket_info, tile_config}};
}

TileConfigMap BuildStaticSpatialConfig(
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info,
    const common::Target& target) {
  if (base_info->spatial_numel == 1) {  // reduce all
    BucketInfo bucket_info{/* sp_lower_bound = */ 1,
                           /* sp_upper_bound = */ 1,
                           /* rb_lower_bound = */ 1,
                           /* rb_upper_bound = */ kMaxNumel,
                           /* sp_is_dynamic = */ false,
                           /* rb_is_dynamic = */ true};
    TileConfig tile_config{/* warp_num = */ 8,
                           /* tree_reduce_num = */ 256,
                           /* grid_reduce_num = */ 1,
                           /* spatial_inner_num = */ 1,
                           BlockReduceMethod()};
    return {{bucket_info, tile_config}};
  } else {
    BucketInfo bucket_info_1_256{/* sp_lower_bound = */ 1,
                                 /* sp_upper_bound = */ kMaxNumel,
                                 /* rb_lower_bound = */ 1,
                                 /* rb_upper_bound = */ 256,
                                 /* sp_is_dynamic = */ false,
                                 /* rb_is_dynamic = */ true};
    TileConfig tile_config_1_256{/* warp_num = */ 8,
                                 /* tree_reduce_num = */ 32,
                                 /* grid_reduce_num = */ 1,
                                 /* spatial_inner_num = */ 1,
                                 WarpReduceMethod()};

    BucketInfo bucket_info_257_2048{/* sp_lower_bound = */ 1,
                                    /* sp_upper_bound = */ kMaxNumel,
                                    /* rb_lower_bound = */ 257,
                                    /* rb_upper_bound = */ 2048,
                                    /* sp_is_dynamic = */ false,
                                    /* rb_is_dynamic = */ true};
    TileConfig tile_config_257_2048{/* warp_num = */ 8,
                                    /* tree_reduce_num = */ 128,
                                    /* grid_reduce_num = */ 1,
                                    /* spatial_inner_num = */ 1,
                                    BlockReduceMethod()};

    BucketInfo bucket_info_2049_INF{/* sp_lower_bound = */ 1,
                                    /* sp_upper_bound = */ kMaxNumel,
                                    /* rb_lower_bound = */ 2049,
                                    /* rb_upper_bound = */ kMaxNumel,
                                    /* sp_is_dynamic = */ false,
                                    /* rb_is_dynamic = */ true};
    TileConfig tile_config_2049_INF{/* warp_num = */ 8,
                                    /* tree_reduce_num = */ 256,
                                    /* grid_reduce_num = */ 1,
                                    /* spatial_inner_num = */ 1,
                                    BlockReduceMethod()};

    return {{bucket_info_1_256, tile_config_1_256},
            {bucket_info_257_2048, tile_config_257_2048},
            {bucket_info_2049_INF, tile_config_2049_INF}};
  }
}

TileConfigMap BuildStaticReduceConfig(
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info,
    const common::Target& target) {
  if (base_info->reduce_numel == 1) {
    BucketInfo bucket_info__1_1023{/* sp_lower_bound = */ 1,
                                   /* sp_upper_bound = */ 1023,
                                   /* rb_lower_bound = */ 1,
                                   /* rb_upper_bound = */ 1,
                                   /* sp_is_dynamic = */ true,
                                   /* rb_is_dynamic = */ false};
    TileConfig tile_config__1_1023{/* warp_num = */ -1,
                                   /* tree_reduce_num = */ 1,
                                   /* grid_reduce_num = */ 1,
                                   /* spatial_inner_num = */ 1,
                                   NoneReduceMethod()};
    BucketInfo bucket_info__1024_1M{/* sp_lower_bound = */ 1024,
                                    /* sp_upper_bound = */ 1024 * 1024 - 1,
                                    /* rb_lower_bound = */ 1,
                                    /* rb_upper_bound = */ 1,
                                    /* sp_is_dynamic = */ true,
                                    /* rb_is_dynamic = */ false};
    TileConfig tile_config__1024_1M{/* warp_num = */ 32,
                                    /* tree_reduce_num = */ 1,
                                    /* grid_reduce_num = */ 1,
                                    /* spatial_inner_num = */ 4,
                                    NoneReduceMethod()};
    BucketInfo bucket_info__1M_INF{/* sp_lower_bound = */ 1024 * 1024,
                                   /* sp_upper_bound = */ kMaxNumel,
                                   /* rb_lower_bound = */ 1,
                                   /* rb_upper_bound = */ 1,
                                   /* sp_is_dynamic = */ true,
                                   /* rb_is_dynamic = */ false};
    TileConfig tile_config__1M_INF{/* warp_num = */ 32,
                                   /* tree_reduce_num = */ 1,
                                   /* grid_reduce_num = */ 1,
                                   /* spatial_inner_num = */ 4,
                                   NoneReduceMethod()};
    return {{bucket_info__1_1023, tile_config__1_1023},
            {bucket_info__1024_1M, tile_config__1024_1M},
            {bucket_info__1M_INF, tile_config__1M_INF}};
  } else if (base_info->reduce_numel <= 256) {
    BucketInfo bucket_info{/* sp_lower_bound = */ 1,
                           /* sp_upper_bound = */ kMaxNumel,
                           /* rb_lower_bound = */ 2,
                           /* rb_upper_bound = */ 256,
                           /* sp_is_dynamic = */ true,
                           /* rb_is_dynamic = */ false};
    TileConfig tile_config{
        /* warp_num = */ 8,
        /* tree_reduce_num = */ 32,
        /* grid_reduce_num = */ 1,
        /* spatial_inner_num = */ (256 / CeilPow2(base_info->reduce_numel)),
        WarpReduceMethod()};
    return {{bucket_info, tile_config}};
  } else if (base_info->reduce_numel <= 2048) {
    int64_t reduce_block =
        int64_t(std::ceil(base_info->reduce_numel * 1.0 / 256.0)) * 256;
    int64_t warp_num = reduce_block / 256;
    int64_t reduce_inner_num = 8;
    int64_t tree_reduce_num = reduce_block / reduce_inner_num;
    BucketInfo bucket_info{/* sp_lower_bound = */ 1,
                           /* sp_upper_bound = */ kMaxNumel,
                           /* rb_lower_bound = */ 257,
                           /* rb_upper_bound = */ 2048,
                           /* sp_is_dynamic = */ true,
                           /* rb_is_dynamic = */ false};
    TileConfig tile_config{warp_num,
                           tree_reduce_num,
                           /* grid_reduce_num = */ 1,
                           /* spatial_inner_num */ 1,
                           BlockReduceMethod()};
    return {{bucket_info, tile_config}};
  } else {
    BucketInfo bucket_info{/* sp_lower_bound = */ 1,
                           /* sp_upper_bound = */ kMaxNumel,
                           /* rb_lower_bound = */ 2049,
                           /* rb_upper_bound = */ kMaxNumel,
                           /* sp_is_dynamic = */ true,
                           /* rb_is_dynamic = */ false};
    TileConfig tile_config{/* warp_num = */ 32,
                           /* tree_reduce_num = */ 1024,
                           /* grid_reduce_num = */ 1,
                           /* spatial_inner_num = */ 1,
                           BlockReduceMethod()};
    return {{bucket_info, tile_config}};
  }
}

TileConfigMap BuildDynamicShapeConfig(
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info,
    const common::Target& target) {
  BucketInfo bucket_info{/* sp_lower_bound = */ 1,
                         /* sp_upper_bound = */ kMaxNumel,
                         /* rb_lower_bound = */ 1,
                         /* rb_upper_bound = */ kMaxNumel,
                         /* sp_is_dynamic = */ true,
                         /* rb_is_dynamic = */ true};
  TileConfig tile_config{/* warp_num = */ 32,
                         /* tree_reduce_num = */ 1024,
                         /* grid_reduce_num = */ 1,
                         /* spatial_inner_num = */ 1,
                         BlockReduceMethod()};
  return {{bucket_info, tile_config}};
}

std::unordered_map<BucketInfo, ScheduleConfig, BucketInfoHash>
CombineBaseInfoAndConfig(
    const TileConfigMap& config_map,
    const std::shared_ptr<ScheduleConfig::BaseInfo>& base_info) {
  std::unordered_map<BucketInfo, ScheduleConfig, BucketInfoHash> combined;
  for (const auto& bucket_config : config_map) {
    ScheduleConfig sch_config{base_info, std::move(bucket_config.second)};
    combined.insert({std::move(bucket_config.first), std::move(sch_config)});
  }
  return combined;
}

std::unordered_map<BucketInfo, ScheduleConfig, BucketInfoHash>
BuildScheduleConfig(const std::shared_ptr<FusionGroupInfo>& group_info,
                    const common::Target& target) {
  std::shared_ptr<ScheduleConfig::BaseInfo> base_info =
      InitBasicInfo(group_info);
  if (!base_info->has_dynamic_reduce && !base_info->has_dynamic_spatial) {
    VLOG(6) << "Building static sptial and static reduce config.";
    return CombineBaseInfoAndConfig(
        BuildPureStaticShapeConfig(base_info, target), base_info);
  } else if (base_info->has_dynamic_reduce && !base_info->has_dynamic_spatial) {
    VLOG(6) << "Building static sptial and dynamic reduce config.";
    return CombineBaseInfoAndConfig(BuildStaticSpatialConfig(base_info, target),
                                    base_info);
  } else if (!base_info->has_dynamic_reduce && base_info->has_dynamic_spatial) {
    VLOG(6) << "Building dynamic sptial and static reduce config.";
    return CombineBaseInfoAndConfig(BuildStaticReduceConfig(base_info, target),
                                    base_info);
  } else {  // (base_info->has_dynamic_reduce && base_info->has_dynamic_spatial)
    VLOG(6) << "Building dynamic spatial and dynamic reduce config.";
    return CombineBaseInfoAndConfig(BuildDynamicShapeConfig(base_info, target),
                                    base_info);
  }
}

}  // namespace ir
}  // namespace cinn
