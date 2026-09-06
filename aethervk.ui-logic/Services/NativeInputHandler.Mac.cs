using System;
#if TARGET_IS_OSX
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AetherVk.Logic.Services.NativeInput;
#endif

namespace AetherVk.Logic.Services;

#if !TARGET_IS_OSX

#pragma warning disable IDE0380
public unsafe class MacNativeInputHandler(IntPtr handle, string handleDescriptor, TraceLevel traceLevel, IUiThreadDispatcher dispatcher, ISchedulerProvider schedulerProvider)
  : NativeInputHandlerBase(handle, handleDescriptor, traceLevel, dispatcher, schedulerProvider)
{
  protected override bool HookEvents()
  {
    return false;
  }

  protected override void UnhookEvents()
  {
  }

  protected override void DoSetSolidColor(byte r, byte g, byte b)
  {
  }
}
#pragma warning restore IDE0380

#else

using unsafe CAtColonSignature = delegate* unmanaged[Cdecl]<nint, nint, byte>;
using unsafe VAtColonAtSignature = delegate* unmanaged[Cdecl]<nint, nint, nint, void>;
using unsafe AtAtColonSignature = delegate* unmanaged[Cdecl]<nint, nint, nint>;
using unsafe VAtColonSignature = delegate* unmanaged[Cdecl]<nint, nint, void>;

/// <summary>
/// To intercept events coming to this `NSView`, we are going to be using ISA Swizzling. This means
/// changing an object's class at runtime by modifying its `isa` pointer (hence at class level, as
/// opposed to method swizzling, in which a method pointer is changed at runtime)
///
/// - Each Objective-C object has an `isa` pointer that points toward the class
/// - When Swizzled, the runtime creates a new, hidden, subclass on the fly
/// - the `isa` pointer is redirected to point to the new class rather than the original class
///
/// Unlike method swizzling, which changes a method pointer in the class object, hence affects all
/// objects, ISA swizzling isolates the changes to one target object.
///
/// This helps us to implement Key-Value Observing (KVO), meaning whenever a property setter is
/// called, the intermediate subclass can intercept this method call and construct an event from it.
/// </summary>
/// <seealso href="https://medium.com/@kumarsuraj19111997/method-swizzling-in-ios-understanding-how-firebase-cloud-messaging-uses-it-2071ebb5090b" />
/// <seealso href="https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/KeyValueObserving/Articles/KVOImplementation.html" />
/// <seealso href="https://stackoverflow.com/questions/38877465/are-method-swizzling-and-isa-swizzling-the-same-thing" />
/// <seealso href="https://medium.com/bitmountn/attributes-of-property-nonatomic-retain-strong-weak-etc-b7ea93a0f772" />
public unsafe class MacNativeInputHandler(IntPtr handle, string handleDescriptor, TraceLevel traceLevel, IUiThreadDispatcher dispatcher, ISchedulerProvider schedulerProvider)
  : NativeInputHandlerBase(handle, handleDescriptor, traceLevel, dispatcher, schedulerProvider)
{
  /// <summary>AOT Safe instance mapping for static unmanaged callbacks</summary>
  private static readonly ConcurrentDictionary<IntPtr, MacNativeInputHandler> s_instances = [];

  // Keeps track of the original class so we can restore it at teardown
  private nint _originalClass = 0;

  protected override bool HookEvents()
  {
    // prevent leak of unmanaged objects: rule of thumb: use autorelease pool on anything which is
    // not executed inside an event loop, which has an active pool (eg keyDown)
    using var pool = new CocoaAutoreleasePool();

    s_instances[_handle] = this;

    // get original class and create the swizzled one
    _originalClass = PInvokeObjC.object_getClass((nint)_handle);
    nint swizzledClass = EnsureSwizzledClass(_originalClass);
    // swap clases for our handle, which should be a NSView (TODO add check)
    PInvokeObjC.object_setClass(_handle, swizzledClass);

    // force window associated to nsview to accept mouse events (required for appkit)
    nint window = PInvokeObjC.nint_objc_msgSend(_handle, PInvokeObjC.GetSelector("window"u8));
    if (window != 0)
    {
      // [window setAcceptsMouseMovedEvents:YES]
      PInvokeObjC.void_objc_msgSend_byte(window, PInvokeObjC.GetSelector("setAcceptsMouseMovedEvents:"u8), 1); // YES
    }
    else return false;

    // [view setWantsLayer:YES] -> enables whether an NSView uses a Core animation CALayer
    // backingstore. If not, you are constrained to the CPU based `drawRect:`.
    // This triggers our swizzled `makeBackingLayer` automatically
    PInvokeObjC.void_objc_msgSend_byte(_handle, PInvokeObjC.GetSelector("setWantsLayer"u8), 1); // YES

    // Manually trigger first scale update to ensure Retina bounds match 1:1
    PInvokeObjC.void_objc_msgSend(_handle, PInvokeObjC.GetSelector("viewDidChangeBackingProperties"u8));

    if (_traceLevel >= TraceLevel.Basic)
      Log(TraceLevel.Basic, $"[macOS] ISA Swizzled NSView {_handle:X}");

    return true;
  }

  protected override void UnhookEvents()
  {
    // restore original NSView class
    if (_originalClass != 0)
    {
      PInvokeObjC.object_setClass(_handle, _originalClass);
    }

    s_instances.TryRemove(_handle, out _);
  }

  protected override void DoSetSolidColor(byte r, byte g, byte b)
  {
    // outside of event loop and calling ObjC -> pool
    using var pool = new CocoaAutoreleasePool();

    nint selLayer = PInvokeObjC.GetSelector("layer"u8);
    nint layer = PInvokeObjC.nint_objc_msgSend(_handle, selLayer);
    nint cgColorRef = PInvokeCoreGraphics.CGColorCreateGenericRGB(r / 255.0, g / 255.0, b / 255.0, 1.0);

    nint selSetBgColor = PInvokeObjC.GetSelector("setBackgroundColor:"u8);
    PInvokeObjC.void_objc_msgSend_nint(layer, selSetBgColor, cgColorRef);

    PInvokeCoreGraphics.CGColorRelease(cgColorRef);
  }

  private static nint EnsureSwizzledClass(nint baseClass)
  {
    // Cache name using a GUID to ensure multiple controls don't collide in the Objective-C Runtime
    byte[] nameBytes = System.Text.Encoding.ASCII.GetBytes($"VulkanNSView_{Guid.NewGuid():N}\0");
    nint newClass = 0;
    fixed (byte* pName = nameBytes)
    {
      newClass = PInvokeObjC.objc_allocateClassPair(baseClass, pName, 0);
    }
    if (newClass == baseClass) return baseClass; // we failed, error handling?

    // Add method implementations using AOT-compatible function pointers
    // assumes 64-bit platform
    fixed (byte* v_at_colon_at = "v@:@"u8) // "v@:@" -> returns void,        takes self, cmd, NSEvent
    fixed (byte* c_at_colon = "c@:"u8)     // "c@:"  -> returns char (bool), takes self, cmd
    fixed (byte* at_at_colon = "@@:"u8)    // "@@:"  -> returns id,          takes self, cmd
    fixed (byte* v_at_colon = "v@:"u8)     // "v@:"  -> returns void,        takes self, cmd
    {
      // --- Vulkan & CAMetalLayer Setup ---
      // Swizzle makeBackingLayer to return CAMetalLayer
      PInvokeObjC.class_addMethod(newClass, PInvokeObjC.GetSelector("makeBackingLayer"u8), &MakeBackingLayer, at_at_colon);
      // Track retina resolution changes
      PInvokeObjC.class_addMethod(newClass, PInvokeObjC.GetSelector("viewDidChangeBackingProperties"u8), &ViewDidChangeBackingProperties, v_at_colon);

      // --- Input Tricks ---
      // Force Top-Left coordinate system, instead of Bottom-Left AppKit's default, to match Vulkan
      // and Avalonia
      PInvokeObjC.class_addMethod(newClass, PInvokeObjC.GetSelector("isFlipped"u8), &AcceptsFirstResponder, c_at_colon);

      // --- Insertion in Responder chain for mouse events ---
      PInvokeObjC.class_addMethod(newClass, PInvokeObjC.GetSelector("acceptsFirstResponder"u8), &AcceptsFirstResponder, c_at_colon);

      // --- Events ---
      PInvokeObjC.class_addMethod(newClass, PInvokeObjC.GetSelector("keyDown:"u8), &KeyDown, v_at_colon_at);
      PInvokeObjC.class_addMethod(newClass, PInvokeObjC.GetSelector("keyUp:"u8), &KeyUp, v_at_colon_at);
      PInvokeObjC.class_addMethod(newClass, PInvokeObjC.GetSelector("mouseDown:"u8), &MouseDown, v_at_colon_at);
      PInvokeObjC.class_addMethod(newClass, PInvokeObjC.GetSelector("mouseUp:"u8), &MouseUp, v_at_colon_at);
      PInvokeObjC.class_addMethod(newClass, PInvokeObjC.GetSelector("rightMouseDown:"u8), &RightMouseDown, v_at_colon_at);
      PInvokeObjC.class_addMethod(newClass, PInvokeObjC.GetSelector("rightMouseUp:"u8), &RightMouseUp, v_at_colon_at);
      PInvokeObjC.class_addMethod(newClass, PInvokeObjC.GetSelector("otherMouseDown:"u8), &OtherMouseDown, v_at_colon_at);
      PInvokeObjC.class_addMethod(newClass, PInvokeObjC.GetSelector("otherMouseUp:"u8), &OtherMouseUp, v_at_colon_at);
      PInvokeObjC.class_addMethod(newClass, PInvokeObjC.GetSelector("mouseMoved:"u8), &MouseMoved, v_at_colon_at);
      PInvokeObjC.class_addMethod(newClass, PInvokeObjC.GetSelector("mouseDragged:"u8), &MouseDragged, v_at_colon_at);
      PInvokeObjC.class_addMethod(newClass, PInvokeObjC.GetSelector("rightMouseDragged:"u8), &RightMouseDragged, v_at_colon_at);
      PInvokeObjC.class_addMethod(newClass, PInvokeObjC.GetSelector("otherMouseDragged:"u8), &OtherMouseDragged, v_at_colon_at);
    }

    // register class in the Objective-C runtime
    PInvokeObjC.objc_registerClassPair(newClass);
    return newClass;
  }


  /// <summary>
  /// Retrieves hte unmanaged pointer to the underlying CAMetalLayer
  /// Pass this directly to VkMetalSurfaceCreateInfoEXT.pLayer when creating a Vulkan VkMetalSurface
  ///
  /// Note: Such aforementioned creation function should be executed from the main thread, together
  /// with this getter. we can't split getter and creation.
  /// </summary>
  public IntPtr MetalLayerPointer
  {
    get => PInvokeObjC.nint_objc_msgSend(_handle, PInvokeObjC.GetSelector("layer"u8));
  }

  #region Swizzled_Implementation

  // used for both acceptsFirstResponder and isFlipped
  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static byte AcceptsFirstResponder(nint self, nint cmd) => 1; // YES

  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static nint MakeBackingLayer(nint self, nint cmd)
  {
    nint metalLayerClass = 0;
    fixed (byte* pCAMetalLayer = "CAMetalLayer"u8)
      metalLayerClass = PInvokeObjC.objc_getClass(pCAMetalLayer); // without having to take QuartzCore!

    nint alloc = PInvokeObjC.nint_objc_msgSend(metalLayerClass, PInvokeObjC.GetSelector("alloc"u8));
    return PInvokeObjC.nint_objc_msgSend(alloc, PInvokeObjC.GetSelector("init"u8));
  }

  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static void ViewDidChangeBackingProperties(nint self, nint cmd)
  {
    // take layer and window from view
    nint layer = PInvokeObjC.nint_objc_msgSend(self, PInvokeObjC.GetSelector("layer"u8));
    if (layer == 0) return; // for some reason setWantsLayer wasn't YES?

    nint window = PInvokeObjC.nint_objc_msgSend(self, PInvokeObjC.GetSelector("window"u8));
    double scale = 1.0;

    // take scaling factor from window and propagate it to the view
    if (window != 0)
    {
      scale = PInvokeObjC.SendDouble(window, PInvokeObjC.GetSelector("backingScaleFactor"u8));
    }

    PInvokeObjC.void_objc_msgSend_double(layer, PInvokeObjC.GetSelector("setContentsScale:"u8), scale);
  }

  #endregion

  #region Unmanaged_Appkit_Callbacks

  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static void KeyDown(nint self, nint cmd, nint nsEvent) => HandleKeyEvent(self, nsEvent, true);

  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static void KeyUp(nint self, nint cmd, nint nsEvent) => HandleKeyEvent(self, nsEvent, false);

  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static void MouseDown(nint self, nint cmd, nint nsEvent) => HandleMouseEvent(self, nsEvent, MouseButton.Left, true);

  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static void MouseUp(nint self, nint cmd, nint nsEvent) => HandleMouseEvent(self, nsEvent, MouseButton.Left, false);

  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static void RightMouseDown(nint self, nint cmd, nint nsEvent) => HandleMouseEvent(self, nsEvent, MouseButton.Right, true);

  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static void RightMouseUp(nint self, nint cmd, nint nsEvent) => HandleMouseEvent(self, nsEvent, MouseButton.Right, false);

  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static void OtherMouseDown(nint self, nint cmd, nint nsEvent) => HandleMouseEvent(self, nsEvent, MouseButton.Middle, true);

  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static void OtherMouseUp(nint self, nint cmd, nint nsEvent) => HandleMouseEvent(self, nsEvent, MouseButton.Middle, false);

  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static void MouseMoved(nint self, nint cmd, nint nsEvent) => HandleMouseEvent(self, nsEvent, MouseButton.None, false);

  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static void MouseDragged(nint self, nint cmd, nint nsEvent) => HandleMouseEvent(self, nsEvent, MouseButton.Left, true);

  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static void RightMouseDragged(nint self, nint cmd, nint nsEvent) => HandleMouseEvent(self, nsEvent, MouseButton.Right, true);

  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static void OtherMouseDragged(nint self, nint cmd, nint nsEvent) => HandleMouseEvent(self, nsEvent, MouseButton.Middle, true);

  #endregion

  private static void HandleKeyEvent(nint self, nint nsEvent, bool isDown)
  {
    if (!s_instances.TryGetValue(self, out var instance)) return;

    ushort keyCode = PInvokeObjC.ushort_objc_msgSend(nsEvent, PInvokeObjC.GetSelector("keyCode"u8));
    ulong modifierFlags = PInvokeObjC.ulong_objc_msgSend(nsEvent, PInvokeObjC.GetSelector("modifierFlags"u8));

    instance.PublishKeyEvent(NormalizeMacKeyCode(keyCode), isDown, ParseModifiers(modifierFlags));
  }

  private static void HandleMouseEvent(nint self, nint nsEvent, MouseButton button, bool isDown)
  {
    if (!s_instances.TryGetValue(self, out var instance)) return;

    // Hover-only motion (no button) is not consumed by any camera mode.
    // mouseMoved: still fires (AppKit requires the override) but we don't publish.
    // Consistent with Windows (WM_MOUSEMOVE without MK_* flags) and Linux (MotionNotify without state).
    if (button == MouseButton.None) return;

    // Get window coordinates with nsEvent.locationInWindow
    PInvokeObjC.CGPoint windowPoint = PInvokeObjC.CGPoint_obj_msgSend(nsEvent, PInvokeObjC.GetSelector("locationInWindow"u8));
    // convert in view coordinates with [view convertPoint:windowPoint fromView:nil]
    PInvokeObjC.CGPoint viewPoint = PInvokeObjC.CGPoint_obj_msgSend_Point_nint(self, PInvokeObjC.GetSelector("convertPoint:fromView:"u8), windowPoint, 0);

    ulong modifierFlags = PInvokeObjC.ulong_objc_msgSend(nsEvent, PInvokeObjC.GetSelector("modifierFlags"u8));

    // Note: AppKit origin (0,0) is on the bottom left, Vulkan and Avalonia Both expect Top-Left, so
    // our view has the isFlipped property set to YES
    instance.PublishMouseEvent(viewPoint.X, viewPoint.Y, button, isDown, ParseModifiers(modifierFlags));
  }

  private static NativeModifierFlags ParseModifiers(ulong flagsRaw)
  {
    NativeModifierFlags m = NativeModifierFlags.None;
    NSEventModifierFlags flags = (NSEventModifierFlags)flagsRaw & NSEventModifierFlags.DeviceIndependentFlagsMask;
    if (flags.HasFlag(NSEventModifierFlags.Shift)) m |= NativeModifierFlags.Shift;
    if (flags.HasFlag(NSEventModifierFlags.Control)) m |= NativeModifierFlags.Control;
    if (flags.HasFlag(NSEventModifierFlags.Option)) m |= NativeModifierFlags.Alt;
    if (flags.HasFlag(NSEventModifierFlags.Command)) m |= NativeModifierFlags.Super;
    return m;
  }

  /// <summary>
  /// Translates macOS physical hardware keyCodes into the unified Win32-style
  /// virtual key standard used by the shared logic dictionary.
  /// </summary>
  private static uint NormalizeMacKeyCode(ushort macKeyCode) => macKeyCode switch
  {
    // --- Alphabet (A-Z) ---
    0x00 => 0x41, // A
    0x0B => 0x42, // B
    0x08 => 0x43, // C
    0x02 => 0x44, // D
    0x0E => 0x45, // E
    0x03 => 0x46, // F
    0x05 => 0x47, // G
    0x04 => 0x48, // H
    0x22 => 0x49, // I
    0x26 => 0x4A, // J
    0x28 => 0x4B, // K
    0x25 => 0x4C, // L
    0x2E => 0x4D, // M
    0x2D => 0x4E, // N
    0x1F => 0x4F, // O
    0x23 => 0x50, // P
    0x0C => 0x51, // Q
    0x0F => 0x52, // R
    0x01 => 0x53, // S
    0x11 => 0x54, // T
    0x20 => 0x55, // U
    0x09 => 0x56, // V
    0x0D => 0x57, // W
    0x07 => 0x58, // X
    0x10 => 0x59, // Y
    0x06 => 0x5A, // Z

    // --- Control Keys ---
    0x35 => 0x1B, // Escape
    0x24 => 0x0D, // Return
    0x4C => 0x0D, // Keypad Enter -> Return
    0x31 => 0x20, // Space

    _ => macKeyCode // Unmapped keys pass through
  };
}

/// <summary>
/// MacOS expects UI Elements to be Objective-C `NSView` objects. Because modern avalonia avoids a
/// dependency on the heavy `Xamarin.Mac`, we need to use P/Invoke into the Objective-C runtime to
/// dynamically subclass `NSView` and intercept its events
///
/// Important Considerations
///
/// - Passing standard delegates to native apple code leads to crashes when .NET GC collects them.
///   We need to go through [UnmanagedCallersOnly] and static function pointers
///
/// - Main Thread Affinity: AppKit is single threaded. NSView manipulation is done through the Main
///   Thread
///
/// - Autorelease pool: When MacOS Calls back to C#, or when we invoke AppKit constructors,
///   intermediate Obj-C objects are generated. We must scope them in an `NSAutoreleasePool` to
///   avoid unmanaged memory leaks
///
/// - Responder Chain: By default, an `NSView` drops keyboard input. We must override
///   `acceptsFirstResponder` to return `YES`
///
/// - On macOS (both x86_64 and ARM64 / Apple Silicon), the native ABI expects the standard C calling convention.
///   CallingConvention defaults to WinApi on windows, and on Cdecl on linux/macos, so on mac it
///   would work. We'll specify it anyways
///
/// - *Note*: on functions which return a pointer, we are assuming we are on a 64-bit system and
///   therefore returning a nint. This would crash on a 32-bit system.
///
///   Fix (unnecessary in out case): add 32 bit overloads, eg `sel_registerName32` and perform an
///   `IntPtr.Size == 8` to choose between the two methods
///
/// </summary>
internal unsafe static class PInvokeObjC
{
  private const string Objc = "/usr/lib/libobjc.A.dylib";

  /// <summary>
  /// Returns the class definition of a specific class
  /// <see href="https://developer.apple.com/documentation/objectivec/objc_getclass(_:)" />
  /// </summary>
  [DllImport(Objc, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern nint objc_getClass(byte* name);

  /// <summary>
  /// <see href="https://developer.apple.com/documentation/objectivec/object_getclass(_:)" />
  /// </summary>
  [DllImport(Objc, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern nint object_getClass(nint obj);

  /// <summary>
  /// <see href="https://developer.apple.com/documentation/objectivec/object_setclass(_:_:)" />
  /// </summary>
  [DllImport(Objc, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern nint object_setClass(nint obj, nint cls);

  /// <summary>
  /// Registers a method with the Objective-C runtime system, maps the method name to a selector,
  /// and returns the selector value
  /// <see href="https://developer.apple.com/documentation/objectivec/sel_registername(_:)" />
  /// </summary>
  [DllImport(Objc, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern nint sel_registerName(byte* name);

  /// <summary>
  /// Creates a new class and metaclass
  /// <see href="https://developer.apple.com/documentation/objectivec/objc_allocateclasspair(_:_:_:)" />
  /// </summary>
  [DllImport(Objc, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern nint objc_allocateClassPair(nint superclass, byte* name, nint extraBytes);

  /// <summary>
  /// Registers a class that was allocated using <see cref="objc_allocateClassPair(nint, byte*, nint)" />
  /// <see href="https://developer.apple.com/documentation/objectivec/objc_registerclasspair(_:)" />
  /// </summary>
  [DllImport(Objc, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void objc_registerClassPair(nint cls);

  /// <summary>
  /// Adds a new method to a class with a given name and implementation
  /// <see href="https://developer.apple.com/documentation/objectivec/class_addmethod(_:_:_:_:)" />
  ///
  /// Note: add as many `class_addMethod` as you need signatures for. See type encodings guide to
  /// know the correct value for each overload https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/ObjCRuntimeGuide/Articles/ocrtTypeEncodings.html#//apple_ref/doc/uid/TP40008048-CH100
  /// </summary>
  [DllImport(Objc, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool class_addMethod(nint cls, nint name, VAtColonAtSignature imp, byte* types);

  [DllImport(Objc, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool class_addMethod(nint cls, nint name, CAtColonSignature imp, byte* types);

  [DllImport(Objc, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool class_addMethod(nint cls, nint name, AtAtColonSignature imp, byte* types);

  [DllImport(Objc, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool class_addMethod(nint cls, nint name, VAtColonSignature imp, byte* types);

  /// <summary>
  /// Sends a message with a sinple return value to an instance of a class
  /// <see href="https://developer.apple.com/documentation/objectivec/objc_msgsend" />
  ///
  /// Note: this method is implemented in raw assembly, and it takes a variadic number of arguments
  /// in Cdecl calling convention and might return any result, following that very same convention,
  /// with the following quirks depending on the CPU architecture (x86_64 vs. ARM64).
  ///
  /// - objc_msgSend_fpret (Floating Point Returns):
  ///    -  x86_64: If the method returns a float or double, you must use objc_msgSend_fpret instead of objc_msgSend.
  ///    -  ARM64 (Apple Silicon): objc_msgSend_fpret does not exist. You just use standard objc_msgSend.
  ///
  /// - objc_msgSend_stret (Struct Returns):
  ///    -  If the method returns a large struct by value (like CGRect), you must use objc_msgSend_stret.
  ///       Note: On ARM64, the rules for when to use _stret are extremely complex and depend on the exact size of the struct.
  ///
  /// Our stategy will be to declare as many overloads of `objc_msgSend` as we need, and since the
  /// return types will always be pointer-like at most, we don't need to worry about these details
  /// </summary>
  [DllImport(Objc, EntryPoint = "objc_msgSend", ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern nint nint_objc_msgSend(nint receiver, nint selector);

  [DllImport(Objc, EntryPoint = "objc_msgSend", ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void void_objc_msgSend(nint receiver, nint selector);

  [DllImport(Objc, EntryPoint = "objc_msgSend", ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void void_objc_msgSend_byte(nint receiver, nint selector, byte arg1);

  [DllImport(Objc, EntryPoint = "objc_msgSend", ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void void_objc_msgSend_nint(nint receiver, nint selector, nint arg1);

  [DllImport(Objc, EntryPoint = "objc_msgSend", ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern ushort ushort_objc_msgSend(nint receiver, nint selector);

  [DllImport(Objc, EntryPoint = "objc_msgSend", ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong ulong_objc_msgSend(nint receiver, nint selector);

  // Struct returning P/Invokes for CGPoint (16 bytes).
  // Note: We safely use the standard objc_msgSend instead of objc_msgSend_stret here.
  //
  // Proof of ABI compliance:
  // 1. x86_64 ABI: Structs <= 16 bytes are returned in registers (XMM0/XMM1 for floats).
  //    Ref: https://developer.apple.com/documentation/xcode/writing-64-bit-intel-code-for-apple-platforms
  // 2. ARM64 ABI: CGPoint is a Homogeneous Floating-point Aggregate (HFA) returned in SIMD registers (d0/d1).
  //    Furthermore, objc_msgSend_stret is completely unused on ARM64.
  //    Ref: https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms
  [DllImport(Objc, EntryPoint = "objc_msgSend", ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern CGPoint CGPoint_obj_msgSend(nint receiver, nint selector);

  [DllImport(Objc, EntryPoint = "objc_msgSend", ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern CGPoint CGPoint_obj_msgSend_Point_nint(nint receiver, nint selector, CGPoint point, nint view);

  [DllImport(Objc, EntryPoint = "objc_msgSend", ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void void_objc_msgSend_double(nint receiver, nint selector, double arg1);

  // not safe to call, cause of the _fpret variant in x86_64 (Note: how can I test it since I don't
  // own an x86_64 mac?)
  [DllImport(Objc, EntryPoint = "objc_msgSend", ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  private static extern double double_objc_msgSend(nint receiver, nint selector);

  // this was actually removed in ARM64 Apple silicon (we are assuming 64 bit here)
  [DllImport(Objc, EntryPoint = "objc_msgSend_fpret", ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  private static extern double double_objc_msgSend_fpret(nint receiver, nint selector);

  /// <summary>
  /// ABI-Safe m to fetch a double (CGFloat) from an Objective-C object.
  /// On Intel (x86_64), floating point returns require objc_msgSend_fpret.
  /// On Apple Silicon (ARM64), standard objc_msgSend is used.
  /// </summary>
  public static double SendDouble(nint receiver, nint selector)
  {
    if (RuntimeInformation.ProcessArchitecture == Architecture.X64)
      return double_objc_msgSend_fpret(receiver, selector);

    return double_objc_msgSend(receiver, selector);
  }

  [StructLayout(LayoutKind.Sequential)]
  internal struct CGPoint
  {
    public double X;
    public double Y;
  }

  /// <summary>Convenient handler for dynamic strings. Delete if not needed</summary>
  public static nint GetClassDynamic(string className)
  {
    if (string.IsNullOrEmpty(className)) return 0;
    // pin the managed string and get a raw pointer out of it
    fixed (char* pChars = className)
    {
      // determine bytes needed for UTF-8
      int byteCount = System.Text.Encoding.UTF8.GetByteCount(pChars, className.Length);
      // Allocate it on the stack (should fit) + null terminator
      byte* buffer = stackalloc byte[byteCount + 1];
      // encode to UTF-8
      System.Text.Encoding.UTF8.GetBytes(pChars, className.Length, buffer, byteCount);
      // null termination
      buffer[byteCount] = 0;

      return objc_getClass(buffer);
    }
  }

  /// <summary>Convenient getter for a selector. ""u8 literals are already null terminated</summary>
  public static nint GetSelector(ReadOnlySpan<byte> nameWithNullTerminator)
  {
    fixed (byte* p = nameWithNullTerminator) return sel_registerName(p);
  }
}

/// <summary>
/// Unmanaged Memory Leak Wrapper
///
/// Intentionally made as a `struct` and not a `class` cause This is not Allocated on the managed
/// heap, but on the stack
/// </summary>
internal unsafe readonly struct CocoaAutoreleasePool : IDisposable
{
  private readonly nint _pool;

  public CocoaAutoreleasePool()
  {
    // Using `u8` suffix gives a ReadOnlySpan<byte> of a UTF-8 string at compile time.
    // `fixed` pins it (which is a no-op since it has a static lifetime) and gives us a `byte*`
    // Null terminator is implicit: https://github.com/dotnet/csharplang/blob/main/proposals/csharp-11.0/utf8-string-literals.md#u8-suffix-on-string-literals
    fixed (byte* pPoolClass = "NSAutoreleasePool"u8)
    fixed (byte* pAlloc = "alloc"u8)
    fixed (byte* pInit = "init"u8)
    {
      nint poolClass = PInvokeObjC.objc_getClass(pPoolClass);
      // NSAutoreleasePool* pool = [NSAutoreleasePool alloc]
      nint allocMsg = PInvokeObjC.nint_objc_msgSend(poolClass, PInvokeObjC.sel_registerName(pAlloc));
      // pool = [pool init]
      _pool = PInvokeObjC.nint_objc_msgSend(allocMsg, PInvokeObjC.sel_registerName(pInit));
    }
  }

  public void Dispose()
  {
    if (_pool != 0)
    {
      fixed (byte* pDrain = "drain"u8)
      {
        // [pool drain]
        PInvokeObjC.void_objc_msgSend(_pool, PInvokeObjC.sel_registerName(pDrain));
      }
    }
  }
}

internal static class PInvokeCoreGraphics
{
  // for Vulkan-less graphics (some color to get started)
  private const string CoreGraphics = "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics";

  /// <summary>
  /// Creates a color in the generic RGB Color Space
  /// <see href="https://developer.apple.com/documentation/coregraphics/cgcolor/init(red:green:blue:alpha:)?language=objc" />
  /// - all inputs are from 0.0 to 1.0
  /// - returns a `CGColorRef` color object (as an id, hence nint on 64-bit machines)
  /// Note: `CGFloat` should be a 64-bit double on all modern, 64-bit, MacOS/Apple platforms
  /// </summary>
  [DllImport(CoreGraphics, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern nint CGColorCreateGenericRGB(double red, double green, double blue, double alpha);

  [DllImport(CoreGraphics, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void CGWarpMouseCursorPosition(PInvokeObjC.CGPoint newCursorPosition);

  /// <summary>
  /// Decrements the retain count of a `CGColorRef`. Functionally equivalient to `CFRelease`,
  /// opposite to `CGColorRetain`
  /// <see href="https://developer.apple.com/documentation/coregraphics/cgcolorrelease" />
  /// </summary>
  [DllImport(CoreGraphics, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void CGColorRelease(nint color);
}

// open the following file (Or wherever you installed you framework)
// /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/AppKit.framework/Headers/NSEvent.h
// to get these raw values. While names can change from a MacOS SDK to another, supposedly,
// numerical values shouldn't change
[Flags]
public enum NSEventModifierFlags : ulong
{
  None = 0,

  // Raw Apple values defined via bitshifts
  CapsLock = 1UL << 16,  // 0x10000   (65,536)
  Shift = 1UL << 17,  // 0x20000   (131,072)
  Control = 1UL << 18,  // 0x40000   (262,144)
  Option = 1UL << 19,  // 0x80000   (524,288) - Also known as Alternate
  Command = 1UL << 20,  // 0x100000  (1,048,576)
  NumericPad = 1UL << 21,  // 0x200000  (2,097,152)
  Help = 1UL << 22,  // 0x400000  (4,194,304)
  Function = 1UL << 23,  // 0x800000  (8,388,608)

  // Used to mask out hardware-dependent bits
  DeviceIndependentFlagsMask = 0xffff0000UL // (4,294,901,760)
}

#endif
