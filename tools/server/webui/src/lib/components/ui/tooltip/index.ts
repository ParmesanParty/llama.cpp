// Local Tooltip implementation backed by the native HTML `popover` attribute
// and top-layer rendering. Replaces bits-ui's TooltipPrimitive in the llm-
// server webui for two reasons:
//
//   1. Flicker fix. bits-ui's Tooltip.Provider coordinated open/close state
//      across sibling Tooltip.Roots via a shared context. Moving the cursor
//      between adjacent triggers would briefly re-emit the source root's
//      open=true on a single frame, producing a visible flash on the source
//      tooltip. Here, each tooltip carries its own pointerenter/leave
//      listener with no shared coordinator, so the bug cannot recur.
//
//   2. Native top-layer rendering. Tooltips use `popover="manual"` and call
//      showPopover()/hidePopover() to render in the browser's top layer.
//      Top-layer rendering escapes every clipping mechanism (overflow,
//      transform, clip-path, contain), so tooltips inside dropdown menus,
//      modals, and other overflow:hidden parents are not cut off — matching
//      what bits-ui's Portal achieved, but native and JS-portal-free.
//
// The exported namespace shape (Root/Trigger/Content/Provider/Portal +
// capitalized aliases) matches bits-ui so existing call sites compile
// unchanged. Provider and Portal become no-op passthroughs.
import Root from './tooltip-root.svelte';
import Trigger from './tooltip-trigger.svelte';
import Content from './tooltip-content.svelte';
import Passthrough from './tooltip-passthrough.svelte';

const Provider = Passthrough;
const Portal = Passthrough;

export {
	Root,
	Trigger,
	Content,
	Provider,
	Portal,
	//
	Root as Tooltip,
	Content as TooltipContent,
	Trigger as TooltipTrigger,
	Provider as TooltipProvider,
	Portal as TooltipPortal
};
