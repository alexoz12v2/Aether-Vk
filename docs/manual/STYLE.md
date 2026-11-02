# AetherVk manual: writing standard

Every panel of the app is documented the same way, in the manner of the Blender manual:
the reader looks up a control and finds what it does, in what unit, what it starts at,
and whether it changes the simulation or only the picture.
`sample-settings-viewport.tex` is the reference implementation of this document. When the
two disagree, fix one of them.

---

## 1. Scope and voice

- **Reference, not tutorial.** One entry per control, findable by its on-screen label.
  Workflows (pick a comet, commit, play) go in a short "Using the panel" paragraph at the
  start of a panel or in the thesis' own chapters, and link to the entries.
- **Present tense, second person, imperative for actions.** "Type the angle and press
  Enter." "The field of view sets ..." Avoid "the user", "we", "will".
- **Say what changes.** Every entry says whether it alters the simulated physics, only the
  rendering, only the camera, or performs an action (§4).
- **Facts from the source, not from memory.** Ranges, defaults and help texts come from the
  places listed in §3.2. A value not yet checked keeps a `% VERIFY` comment on its line.
- **No person's name** in the text, comments, figure files or metadata.

## 2. Chapter order

Follow the order in which a user meets the app (the panel keys are in `inventory.md`):

1. Main window: menus (`menu-file`, `menu-edit`), mode bar, docking (`main`, `docking`).
2. Comet tab (`comet`): guided steps, selection, simulated range, commit, nucleus, rotation.
3. Timeline tab (`timeline`).
4. Model tab (`model`): Sun, nucleus spin, dust and jets, defaults.
5. Settings tab (`settings`): time scale, general, orbital elements, comet appearance,
   billboards and FITS, particle rendering, input bindings, viewport.
6. Imports tab (`imports`) and its dialogs.
7. Viewport and overlay (`viewport`, `overlay`, `billboard-window`).
8. Dialogs (`dlg-*`), help flyout, splash. Debug panels go in an appendix, if at all.

One `.tex` file per panel: `panels/<panel-key>.tex`, `\input` from the chapter.

## 3. The panel template

### 3.1 Skeleton

```latex
\begin{uipanel}[key=settings-viewport,
                open={\menu{Settings tab > Viewport}},
                modes={Orbit, Nucleus, Model; some rows depend on the mode (table~\ref{tab:...})}]
               {Viewport}
% 1. purpose: 2-4 sentences, what the panel is for and when to use it
% 2. how to open: in the info box (open=), plus anything else needed (a committed comet, ...)
% 3. screenshot(s) with callouts
\begin{figure}[htbp]\centering
  \uiscreenshot{settings-viewport}{img/settings-viewport--orbit--light.png}
  \caption{The \uilabel{Viewport} section in Orbit view.}\label{fig:settings-viewport}
\end{figure}
% 4. sections, in on-screen order, top to bottom
\uisection[projection]{Projection}{ProjectionAdvancedExpander}
\begin{uiproperty}[key=fov, id=FovSlider, ...]{FOV} ... \end{uiproperty}
...
\end{uipanel}
```

Rules:

- Sections follow the expanders of the panel, in on-screen order. A block without an
  expander gets a `\uisection` with an empty id argument.
- Properties follow the on-screen order inside a section: top to bottom, then left to right.
- A control that appears only in some states (perspective vs orthographic, Model view only,
  a jet selected) gets its own figure of that state, and `visible=` in its entry.
- Group trivially related controls (a value plus its unit picker; Lat/Lon) into one entry
  with one label, and list both ids in the text.

### 3.2 The property entry

```latex
\begin{uiproperty}[key=fov, id=FovSlider, type=slider and number box,
    unit={\unit{\degree}, \unit{\arcminute} or \unit{\arcsecond} (\uilabel{FovUnitPicker})},
    range={\ang{;;0.02} to \ang{179}}, default={\ang{30}}, % VERIFY
    effect=camera, saved=file, visible={Perspective projection},
    help=Tabs_Settings_HelpViewFov]{FOV}
What it does, 1-3 sentences.
\begin{uihelptext} <the in-app help text, verbatim> \end{uihelptext}
Notes: interactions, edge cases, units of the readout.
\begin{uiexample} A typical value, typed as the user would type it. \end{uiexample}
\end{uiproperty}
```

| Field | Content | Source of truth |
|---|---|---|
| label (argument) | the on-screen label, exactly, without the trailing colon | resx English value via `inventory.md` |
| `key` | short local key, lowercase, hyphens: label `prop:<panel>.<key>` | chosen once, never renamed |
| `id` | AutomationId; also selects the callout | `inventory.md` |
| `type` | one word from the type vocabulary below | the axaml control |
| `unit` | siunitx unit(s); name the unit picker if there is one | the axaml / VM converter |
| `range` | min to max, plus step if there is one ("step 0.5") | VM clamp constants, `Minimum`/`Maximum` in axaml, engine validation |
| `default` | value on a fresh launch / new save | VM initializer, `*Dto` default, `PreferencesService` default |
| `effect` | `physics`, `display`, `camera`, `action` (§4) | judgement, from what the VM calls |
| `effectnote` | one line, only if the badge is not enough ("requires a recommit") | |
| `saved` | `file` (in the save, `SaveIndexV1` DTOs), `prefs` (`~/.aethervk/preferences.json`), `none` | DTOs / `PreferencesService` |
| `visible` | the state in which the control exists, if not always | axaml `IsVisible` bindings |
| `help` | resx key of the `?` help text | `PropertyHelpButton` `HelpContent` |

Type vocabulary: `check box`, `toggle`, `radio buttons`, `drop-down`, `number box`,
`slider`, `slider and number box`, `text box`, `search box`, `button`, `list`, `table`,
`date/time box`, `read-out` (text that only shows a value), `colour picker`, `file picker`.

Order of the body: what it does → help text → notes → example. Omit what is empty. Do not
repeat the help text in your own words; add only what it does not say (units of read-outs,
side effects, persistence, interplay with other controls, with `\propref` links).

### 3.3 Read-outs and actions

- A read-out (status line, scale text) gets an entry with `type=read-out` and no range or
  default. Show a real sample in `uiexample`, copied from a screenshot.
- A button gets `effect=action`. Say what happens, what it needs first (a committed comet),
  and whether it can be undone.

## 4. Effect categories

| `effect=` | Badge | Use when the control ... | Example |
|---|---|---|---|
| `physics` | Physics | changes what is simulated: orbit, activity, grains, time range. Results differ | jet activity, grain size, simulated range |
| `display` | Display | changes how results are drawn, not the results | exposure, orbit lines, star background |
| `camera` | Camera | moves or reshapes the view (projection, observer, look direction) | FOV, near/far, topocentric |
| `action` | Action | runs something once | Commit, View nucleus from Earth, Snap above |

If a control changes physics only after a recommit, use `physics` and say so in
`effectnote`.

## 5. Terminology

| Use | Not | Meaning |
|---|---|---|
| **Orbit view**, **Nucleus view**, **Model view** | UpZenith, CometOrbiting, EarthPosition | the three view modes of the mode bar. The enum names appear only in a footnote for developers |
| camera mode | view mode | the read-out `CameraModeText`; use "view mode" for the three buttons |
| **commit** / **recommit** | apply, run, regenerate | compute the simulation for the chosen comet and range / again after a change |
| **range lock** | | the state after a range edit until recommit (`ModeBarRangeLockText`) |
| epoch | time, date | an instant; always give the time scale: `2021-12-28 00:00 TDB` |
| TDB, UTC | | as on screen; never mix them in one example |
| billboard | image, sprite | an image placed in the scene (e.g. a FITS frame) |
| jet | source, emitter | an active region on the nucleus |
| grain, dust | particle (except in *Particle rendering*) | |
| AU | au, UA | as the UI writes it |
| **check** / **clear** | tick, enable/disable | for check boxes |
| **expand** / **collapse** | open/close | for expanders |
| **choose** *X* from *Y* | select | for drop-downs and radio buttons |
| **type** *v* **and press** \uikey{Enter} | enter, input | for number boxes (they commit on Enter or focus loss) |
| click, double-click, drag, scroll | press (for mouse) | |
| tab, pane, section, panel | | tab: a docking tab; pane: the docked frame holding tabs; section: an expander; panel: one documented unit (a tab or a section of it) |

## 6. Typography and macros

Load after the thesis' `commands.tex`: `\usepackage[english]{aethervk-manual}` (options:
`italian`, `menukeys`, `nouifont`, `placeholders`).

| Macro | For | Example |
|---|---|---|
| `\uilabel{FOV}` | any on-screen text: labels, buttons, tab and section names | Click \uilabel{Commit}. |
| `\uival{0.0001}` | a value as typed or shown in a box | type \uival{34.1} |
| `\uiid{FovSlider}` | an AutomationId (small, monospace, grey) | |
| `\resx{Tabs_Settings_HelpViewFov}` | a resource key | |
| `\menu{File > Save As...}` | a menu path; `>` separates the items | |
| `\uikey{Ctrl+S}` | a key or chord | press \uikey{Enter} |
| `\uimouse{Wheel}` | a mouse action | |
| `\unit{\AU}`, `\qty{0.5}{\EV}`, `\ang{30}` | units and quantities: siunitx (v2 in the thesis: `\unit`/`\qty` are aliased to `\si`/`\SI`) | |
| `\callout{3}` | a marker number in running text | |
| `\co{FovSlider}` | the marker of an id in the current figure (`\co[fig]{id}` for another) | the slider \co{FovSlider} |
| `\uiscreenshot[opts]{fig}{file}` | a screenshot with its callouts | |
| `\uiusefigure{fig}` | which figure the following entries take their markers from | |
| `uipanel`, `\uisection`, `uiproperty`, `uihelptext`, `uiexample` | the structure (§3) | |
| `\panelref`, `\sectionref`, `\propref` | cross-references (§7) | |

Rules:

- Never write raw `\textbf` / `\texttt` for UI text. The macros keep fonts and colours
  consistent, and can be restyled later in one place.
- Units: always through siunitx; AU as the unit `\AU`, pixels as `\pixel`, exposure as
  `\EV`. Angles with `\ang` (`\ang{;34.1;}` for 34.1′). A sign that must show needs
  `retain-explicit-plus` (siunitx v2 drops "+" otherwise): `\num[retain-explicit-plus]{+3}`.
- Numbers in running text follow the chapter language. Numbers inside `\uival` follow the
  screenshots (decimal comma if they were captured under it-IT, see PLAN §2.2).
- Dates: ISO 8601, with the time scale: `2021-12-12 00:00 UTC`.
- Help texts in `uihelptext` are quoted verbatim, including their punctuation; do not fix
  them in the manual, fix them in the resx.
- The `.sty` does not use the tcolorbox `skins` library (it needs `tikzfill`, absent from
  some installs), does not redefine the thesis' `\file` or `\code`, and loads siunitx only
  if it is not loaded yet.

## 7. Labels, keys, files, cross-references

| Thing | Pattern | Example |
|---|---|---|
| panel | `panel:<panel-key>` (set by `uipanel[key=]`) | `panel:settings-viewport` |
| section | `panelsec:<panel-key>/<label>` (set by `\uisection[label]`) | `panelsec:settings-viewport/projection` |
| property | `prop:<panel-key>.<key>` (set by `uiproperty[key=]`) | `prop:settings-viewport.fov` |
| figure | `fig:<figure-key>` | `fig:settings-viewport-persp` |
| figure key | panel key, plus `-<part>` for extra figures | `settings-viewport-persp` |
| screenshot | `img/<figure-key>--<state>--<theme>.png` | `img/settings-viewport--orbit--light.png` |
| callouts | `img/<figure-key>--<state>--<theme>.callouts.tex` (generated) | |
| panel source | `panels/<panel-key>.tex` | |

Panel keys: lowercase ASCII, hyphens, tab name first (`comet-simulated-range`,
`model-dust-jet`). They are permanent; rename only with a global search.

Cross-references:

- To a panel: `\panelref{settings-viewport}` → "Viewport (§ 4.3)".
- To a section: `\sectionref{settings-viewport/projection}`.
- To a control: `\propref{settings-viewport.fov}` → "**FOV** (p. 37)". Use it whenever an
  entry mentions another control, the first time per entry.
- From the thesis' scientific chapters to the manual: `\propref`, so a reader of the method
  can find the control that sets each parameter.

## 8. Screenshots and captions

- Captured by the `ManualCapture` fixture only (PLAN §2), never by hand: hand-made
  screenshots drift in size, theme and DPI.
- Light theme in the printed thesis; dark only for slides.
- Natural size (150 dpi in the PNG, so 1 UI px = 0.169 mm). Do not scale with `width=`
  except to fit `\linewidth`, which `\uiscreenshot` does by itself. Side-by-side figures
  use `minipage[b]` (bottom-aligned).
- Callouts numbered top to bottom, then left to right; **at most 12 per figure**. More
  means the figure should be split.
- Each callout number appears in the text: as the marker in a property header (automatic,
  from `id=`), or as `\co{id}` in prose.
- Caption: one sentence, naming the section and the state: "The **Projection** section in
  Orbit view, perspective projection." No "Screenshot of".
- Every figure is referenced from the text before it appears.

## 9. Per-panel checklist

- [ ] `uipanel` with `key`, `open`, `modes`; purpose in 2–4 sentences.
- [ ] One figure per state that changes the visible controls; callouts ≤ 12; captions as §8.
- [ ] One `\uisection` per expander, in on-screen order, with its AutomationId.
- [ ] Every AutomationId of the panel in `inventory.md` appears in an `id=` or `\co{}`, or
      is listed in `tools/coverage-allow.txt`.
- [ ] Each entry: type, unit, range, default, effect, saved, help key; `visible` where needed.
- [ ] Help texts verbatim from the resx; each `% VERIFY` resolved or reported.
- [ ] At least one example for every number box with a unit.
- [ ] `\propref` for every other control mentioned.
- [ ] Compiles without `??` (missing labels or callouts) or callout warnings in the log.
