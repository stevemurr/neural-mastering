#include "param_id.hpp"

namespace nablafx {

uint32_t param_id_for(const std::string& effect_name, const std::string& control_id) {
    // FNV-1a 32-bit
    constexpr uint32_t FNV_OFFSET = 0x811C9DC5u;
    constexpr uint32_t FNV_PRIME  = 0x01000193u;
    uint32_t h = FNV_OFFSET;
    auto mix = [&](const std::string& s) {
        for (unsigned char c : s) {
            h ^= c;
            h *= FNV_PRIME;
        }
    };
    mix(effect_name);
    mix(std::string{":"});
    mix(control_id);
    // CLAP reserves param id 0xFFFFFFFF (CLAP_INVALID_ID); map that one off.
    if (h == 0xFFFFFFFFu) h = 0xFFFFFFFEu;
    return h;
}

}  // namespace nablafx
