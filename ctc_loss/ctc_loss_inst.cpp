// Copyright © 2024 Yury Popov (@djphoenix).

#include "ctc_loss/ctc_loss.h"

namespace ctc_ext {

mx::array ctc_loss(
  const mx::array& log_probs,
  const mx::array& targets,
  const mx::array& input_lengths,
  const mx::array& target_lengths,
  mx::uint64_t blank,
  mx::StreamOrDevice s
) {
  auto out_dtype         = log_probs.dtype();
  auto input_time_size   = log_probs.shape()[0];
  auto batch_size        = log_probs.shape()[1];
  auto input_target_size = targets.shape()[1];

  // Output: loss, log_alpha
  return mx::array::make_arrays(
    { { batch_size }, { input_time_size, batch_size, input_target_size * 2 + 2 } },
    { out_dtype, out_dtype },
    std::make_shared<CTCLoss>(to_stream(s), blank),
    { log_probs, targets, input_lengths, target_lengths }
  )[0];
}

std::vector<mx::array> CTCLoss::vjp(
  const std::vector<mx::array>& primals,
  const std::vector<mx::array>& cotangents,
  const std::vector<mx::int>  & argnums,
  const std::vector<mx::array>& outputs
) {
  auto &log_probs      = primals[0];
  auto &targets        = primals[1];
  auto &input_lengths  = primals[2];
  auto &target_lengths = primals[3];
  auto &nll            = outputs[0];
  auto &log_alpha      = outputs[1];
  auto &ctg            = cotangents[0];

  return { array(
    log_probs.shape(), log_probs.dtype(),
    std::make_shared<CTCLossVJP>(stream(), blank_),
    { log_probs, targets, input_lengths, target_lengths, log_alpha, nll, ctg }
  ) };
}

} // namespace ctc_ext
