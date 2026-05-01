#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// Opaque handle – defined only inside tone_gui.mm
typedef struct ToneGUIState ToneGUIState;

// Describes one parameter for the initial handshake.
//
// `enum_options` lists discrete option labels (one per integer value from
// `min` to `max`) for enum-style controls, in the order they should appear in
// the picker. NULL or `n_enum_options == 0` means the control is a continuous
// knob/toggle. Used by the AUTO-EQ class picker (CLS).
typedef struct {
    const char*         id;
    const char*         name;
    float               min;
    float               max;
    float               def;
    const char*         unit;
    float               current_value;
    const char* const*  enum_options;
    int                 n_enum_options;
} ToneParamInfo;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/**
 * Allocate GUI state.  All Objective-C/WKWebView objects are created here.
 *
 * @param plugin_ptr        Opaque pointer passed back verbatim to the two callbacks.
 * @param resources_dir     Absolute path to the bundle's Resources directory.
 *                          The file "ui/index.html" is expected relative to it.
 * @param on_param_change   Called whenever the web UI moves a knob/toggle.
 * @param on_order_change   Called when the user reorders processor cards.
 */
ToneGUIState* tone_gui_create(
    void*        plugin_ptr,
    const char*  resources_dir,
    void       (*on_param_change)(void* plug, const char* param_id, float value),
    void       (*on_order_change)(void* plug, const int* order, int count)
);

/** Release all resources. Safe to call with NULL. */
void tone_gui_destroy(ToneGUIState* gui);

// ---------------------------------------------------------------------------
// Windowing
// ---------------------------------------------------------------------------

/**
 * Embed the GUI into an existing NSView.
 * Must be called before tone_gui_show().
 * @param ns_view_ptr  An NSView* cast to void*.
 * @return true on success.
 */
bool tone_gui_set_parent(ToneGUIState* gui, void* ns_view_ptr);

void tone_gui_show(ToneGUIState* gui);
void tone_gui_hide(ToneGUIState* gui);

/**
 * Returns the fixed pixel dimensions of the GUI window.
 * Safe to call before tone_gui_create().
 */
void tone_gui_get_size(uint32_t* w, uint32_t* h);

// ---------------------------------------------------------------------------
// State sync  (C++ → WebView)
// ---------------------------------------------------------------------------

/**
 * Push the full initial state to the WebView.
 * May be called before the page has finished loading; the implementation
 * buffers the call and replays it in webView:didFinishNavigation:.
 *
 * @param params       Array of ToneParamInfo, one entry per parameter.
 * @param n_params     Length of params[].
 * @param order        Processor order array (values 0-5).
 * @param order_count  Length of order[].
 */
void tone_gui_send_init(
    ToneGUIState*       gui,
    const ToneParamInfo* params,
    int                  n_params,
    const int*           order,
    int                  order_count
);

/**
 * Notify the WebView that a single parameter changed (DAW automation, etc.).
 * Thread-safe: dispatches to main queue internally if needed.
 */
void tone_gui_notify_param(ToneGUIState* gui, const char* param_id, float value);

/**
 * Evaluate arbitrary JavaScript in the WebView.
 * Must be called from the main thread.
 */
void tone_gui_eval_js(ToneGUIState* gui, const char* js);

#ifdef __cplusplus
} // extern "C"
#endif
