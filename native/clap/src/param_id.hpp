#pragma once

#include <cstdint>
#include <string>

namespace nablafx {

// Deterministic 32-bit parameter id derived from (effect_name, control id).
// Hosts persist parameter automation keyed by this id, so it must be stable
// across rebuilds and machines. std::hash is not — FNV-1a is.
uint32_t param_id_for(const std::string& effect_name, const std::string& control_id);

}  // namespace nablafx
