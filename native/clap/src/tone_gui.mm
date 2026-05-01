// tone_gui.mm
// Objective-C++ implementation of the TONE CLAP plugin GUI bridge.
// macOS arm64, ARC enabled.

#import <Cocoa/Cocoa.h>
#import <WebKit/WebKit.h>
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>

#include "tone_gui.h"

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------

@interface ToneMessageHandler : NSObject <WKScriptMessageHandler>
@end

// ---------------------------------------------------------------------------
// Internal C++ struct (not exposed in the header)
// ---------------------------------------------------------------------------

struct ToneGUIState {
    // Callbacks into the plugin
    void*       plugin_ptr           = nullptr;
    void      (*on_param_change)(void*, const char*, float) = nullptr;
    void      (*on_order_change)(void*, const int*, int)    = nullptr;

    // Resources path (copy)
    std::string resources_dir;

    // ObjC objects kept alive under ARC via __strong
    __strong NSView*             container_view  = nil;
    __strong WKWebView*          web_view        = nil;
    __strong ToneMessageHandler* msg_handler     = nil;
    __strong WKWebViewConfiguration* wk_config   = nil;

    // Navigation delegate (holds weak back-ref to this struct)
    __strong id<WKNavigationDelegate> nav_delegate = nil;

    // Buffered JS to run once the page has finished loading
    std::string pending_init_js;
    bool        page_loaded = false;
};

// ---------------------------------------------------------------------------
// Script message handler  (JS → C++)
// ---------------------------------------------------------------------------

@interface ToneMessageHandler () {
    ToneGUIState* _state;
}
- (instancetype)initWithState:(ToneGUIState*)state;
@end

@implementation ToneMessageHandler

- (instancetype)initWithState:(ToneGUIState*)state {
    self = [super init];
    if (self) { _state = state; }
    return self;
}

- (void)userContentController:(WKUserContentController*)controller
      didReceiveScriptMessage:(WKScriptMessage*)message {
    if (!_state) return;

    // message.body should be a NSDictionary
    NSDictionary* body = message.body;
    if (![body isKindOfClass:[NSDictionary class]]) return;

    NSString* type = body[@"type"];
    if (!type) return;

    if ([type isEqualToString:@"setParam"]) {
        NSString* paramId = body[@"id"];
        NSNumber* value   = body[@"value"];
        if (paramId && value && _state->on_param_change) {
            _state->on_param_change(
                _state->plugin_ptr,
                [paramId UTF8String],
                [value floatValue]
            );
        }
    } else if ([type isEqualToString:@"reorder"]) {
        NSArray* orderArr = body[@"order"];
        if ([orderArr isKindOfClass:[NSArray class]] && _state->on_order_change) {
            NSUInteger count = orderArr.count;
            std::vector<int> order(count);
            for (NSUInteger i = 0; i < count; ++i) {
                order[i] = [orderArr[i] intValue];
            }
            _state->on_order_change(_state->plugin_ptr, order.data(), (int)count);
        }
    }
}

@end

// ---------------------------------------------------------------------------
// Navigation delegate  (page-loaded notification)
// ---------------------------------------------------------------------------

@interface ToneNavDelegate : NSObject <WKNavigationDelegate> {
    ToneGUIState* _state;
}
- (instancetype)initWithState:(ToneGUIState*)state;
@end

@implementation ToneNavDelegate

- (instancetype)initWithState:(ToneGUIState*)state {
    self = [super init];
    if (self) { _state = state; }
    return self;
}

- (void)webView:(WKWebView*)webView didFinishNavigation:(WKNavigation*)navigation {
    if (!_state) return;
    _state->page_loaded = true;

    if (!_state->pending_init_js.empty()) {
        NSString* js = [NSString stringWithUTF8String:_state->pending_init_js.c_str()];
        _state->pending_init_js.clear();
        [webView evaluateJavaScript:js completionHandler:nil];
    }
}

@end

// ---------------------------------------------------------------------------
// Helper: read a file into std::string
// ---------------------------------------------------------------------------

static std::string read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return {}; }
    std::string buf(sz, '\0');
    fread(buf.data(), 1, sz, f);
    fclose(f);
    return buf;
}

// ---------------------------------------------------------------------------
// Helper: JSON-escape a C string
// ---------------------------------------------------------------------------

static std::string json_escape(const char* s) {
    std::string out;
    out.reserve(64);
    out += '"';
    for (const char* p = s; *p; ++p) {
        switch (*p) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += *p;     break;
        }
    }
    out += '"';
    return out;
}

// ---------------------------------------------------------------------------
// Public C API implementation
// ---------------------------------------------------------------------------

ToneGUIState* tone_gui_create(
    void*       plugin_ptr,
    const char* resources_dir,
    void      (*on_param_change)(void*, const char*, float),
    void      (*on_order_change)(void*, const int*, int)
) {
    // Must run on the main thread
    if (!NSThread.isMainThread) {
        __block ToneGUIState* result = nullptr;
        dispatch_sync(dispatch_get_main_queue(), ^{
            result = tone_gui_create(plugin_ptr, resources_dir,
                                     on_param_change, on_order_change);
        });
        return result;
    }

    ToneGUIState* state = new ToneGUIState();
    state->plugin_ptr       = plugin_ptr;
    state->on_param_change  = on_param_change;
    state->on_order_change  = on_order_change;
    state->resources_dir    = resources_dir ? resources_dir : "";

    // --- WKWebView configuration ---
    WKWebViewConfiguration* config = [[WKWebViewConfiguration alloc] init];
    WKUserContentController* ucc   = [[WKUserContentController alloc] init];

    ToneMessageHandler* handler = [[ToneMessageHandler alloc] initWithState:state];
    [ucc addScriptMessageHandler:handler name:@"tone"];
    config.userContentController = ucc;

    // Allow JavaScript (macOS 11+ API; suppress deprecation on older SDKs).
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    config.preferences.javaScriptEnabled = YES;
#pragma clang diagnostic pop

    state->wk_config  = config;
    state->msg_handler = handler;

    // --- Container NSView ---
    NSRect frame = NSMakeRect(0, 0, 700, 560);
    NSView* container = [[NSView alloc] initWithFrame:frame];
    container.wantsLayer = YES;
    container.layer.backgroundColor = [[NSColor colorWithSRGBRed:0.067
                                                           green:0.067
                                                            blue:0.067
                                                           alpha:1.0] CGColor];
    state->container_view = container;

    // --- WKWebView ---
    WKWebView* wv = [[WKWebView alloc] initWithFrame:frame configuration:config];
    wv.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    [container addSubview:wv];
    state->web_view = wv;

    // --- Navigation delegate ---
    ToneNavDelegate* navDel = [[ToneNavDelegate alloc] initWithState:state];
    wv.navigationDelegate   = navDel;
    state->nav_delegate     = navDel;

    // --- Load HTML ---
    std::string html_path = state->resources_dir + "/ui/index.html";
    std::string html      = read_file(html_path);

    if (!html.empty()) {
        NSString* htmlStr  = [NSString stringWithUTF8String:html.c_str()];
        // Use the ui/ directory as baseURL so relative asset paths resolve
        NSString* basePath = [NSString stringWithUTF8String:
                              (state->resources_dir + "/ui/").c_str()];
        NSURL*    baseURL  = [NSURL fileURLWithPath:basePath isDirectory:YES];
        [wv loadHTMLString:htmlStr baseURL:baseURL];
    } else {
        // Fallback: show an error page
        NSString* fallback = @"<html><body style='background:#111;color:#f66;"
                              "font-family:monospace;padding:20px'>"
                              "<h2>NeuralMastering GUI</h2>"
                              "<p>Could not load index.html from resources.</p>"
                              "</body></html>";
        [wv loadHTMLString:fallback baseURL:nil];
    }

    return state;
}

void tone_gui_destroy(ToneGUIState* gui) {
    if (!gui) return;

    auto destroy_block = ^{
        if (gui->web_view) {
            [gui->web_view.configuration.userContentController
             removeScriptMessageHandlerForName:@"tone"];
            [gui->web_view removeFromSuperview];
            gui->web_view = nil;
        }
        if (gui->container_view) {
            [gui->container_view removeFromSuperview];
            gui->container_view = nil;
        }
        gui->msg_handler  = nil;
        gui->nav_delegate = nil;
        gui->wk_config    = nil;
    };

    if (NSThread.isMainThread) {
        destroy_block();
    } else {
        dispatch_sync(dispatch_get_main_queue(), destroy_block);
    }

    delete gui;
}

bool tone_gui_set_parent(ToneGUIState* gui, void* ns_view_ptr) {
    if (!gui || !ns_view_ptr) return false;

    auto set_parent_block = ^bool{
        NSView* parent = (__bridge NSView*)ns_view_ptr;
        if (!parent) return false;

        NSRect bounds = parent.bounds;
        gui->container_view.frame = bounds;
        gui->container_view.autoresizingMask =
            NSViewWidthSizable | NSViewHeightSizable;

        [parent addSubview:gui->container_view];
        return true;
    };

    if (NSThread.isMainThread) {
        return set_parent_block();
    } else {
        __block bool result = false;
        dispatch_sync(dispatch_get_main_queue(), ^{ result = set_parent_block(); });
        return result;
    }
}

void tone_gui_show(ToneGUIState* gui) {
    if (!gui) return;
    auto blk = ^{ if (gui->container_view) gui->container_view.hidden = NO; };
    if (NSThread.isMainThread) blk();
    else dispatch_async(dispatch_get_main_queue(), blk);
}

void tone_gui_hide(ToneGUIState* gui) {
    if (!gui) return;
    auto blk = ^{ if (gui->container_view) gui->container_view.hidden = YES; };
    if (NSThread.isMainThread) blk();
    else dispatch_async(dispatch_get_main_queue(), blk);
}

void tone_gui_get_size(uint32_t* w, uint32_t* h) {
    if (w) *w = 700;
    if (h) *h = 560;
}

void tone_gui_send_init(
    ToneGUIState*        gui,
    const ToneParamInfo* params,
    int                  n_params,
    const int*           order,
    int                  order_count
) {
    if (!gui) return;

    // Build JSON: { paramMeta:{...}, paramValues:{...}, processorOrder:[...] }
    std::string js;
    js.reserve(2048);
    js += "toneInit({\"paramMeta\":{";

    for (int i = 0; i < n_params; ++i) {
        const ToneParamInfo& p = params[i];
        if (i) js += ',';
        js += json_escape(p.id);
        js += ":{\"name\":";
        js += json_escape(p.name);

        char buf[128];
        snprintf(buf, sizeof(buf),
                 ",\"min\":%g,\"max\":%g,\"def\":%g,\"unit\":",
                 (double)p.min, (double)p.max, (double)p.def);
        js += buf;
        js += json_escape(p.unit);

        if (p.enum_options && p.n_enum_options > 0) {
            js += ",\"enumOptions\":[";
            for (int k = 0; k < p.n_enum_options; ++k) {
                if (k) js += ',';
                js += json_escape(p.enum_options[k] ? p.enum_options[k] : "");
            }
            js += ']';
        }
        js += '}';
    }

    js += "},\"paramValues\":{";
    for (int i = 0; i < n_params; ++i) {
        const ToneParamInfo& p = params[i];
        if (i) js += ',';
        js += json_escape(p.id);
        char buf[64];
        snprintf(buf, sizeof(buf), ":%g", (double)p.current_value);
        js += buf;
    }

    js += "},\"processorOrder\":[";
    for (int i = 0; i < order_count; ++i) {
        if (i) js += ',';
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", order[i]);
        js += buf;
    }
    js += "]});";

    auto eval_block = ^{
        if (!gui->page_loaded) {
            gui->pending_init_js = js;
        } else {
            NSString* nsjs = [NSString stringWithUTF8String:js.c_str()];
            [gui->web_view evaluateJavaScript:nsjs completionHandler:nil];
        }
    };

    if (NSThread.isMainThread) {
        eval_block();
    } else {
        dispatch_async(dispatch_get_main_queue(), eval_block);
    }
}

void tone_gui_eval_js(ToneGUIState* gui, const char* js) {
    if (!gui || !js) return;
    if (!NSThread.isMainThread) return;  // caller must be on main thread
    if (!gui->web_view || !gui->page_loaded) return;
    NSString* nsjs = [NSString stringWithUTF8String:js];
    [gui->web_view evaluateJavaScript:nsjs completionHandler:nil];
}

void tone_gui_notify_param(ToneGUIState* gui, const char* param_id, float value) {
    if (!gui || !param_id) return;

    std::string id_str = param_id;
    float       val    = value;

    auto blk = ^{
        if (!gui->web_view || !gui->page_loaded) return;
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "toneSetParam(%s,%g);",
                 json_escape(id_str.c_str()).c_str(),
                 (double)val);
        NSString* js = [NSString stringWithUTF8String:buf];
        [gui->web_view evaluateJavaScript:js completionHandler:nil];
    };

    if (NSThread.isMainThread) blk();
    else dispatch_async(dispatch_get_main_queue(), blk);
}
