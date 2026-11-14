-- The Last Guardian PC port (GPL-3.0-or-later): the PC Settings pages,
-- drawn and run by the game itself with its own menu textures and GUI
-- calls (appended to system/scripts/utils.lua; pc_menu.rs). The settings,
-- their values and what applies them live in the port (pc_settings.rs);
-- this script asks for them through GaPrint (`call`).
TLGPC = TLGPC or {}
local P = TLGPC
local floor = math.floor
-- (The port answers with one string, its values joined by \31: one push
-- on the C side, which must not grow this stack; unpack does that here.)
local unpack = unpack or table.unpack
local function call(...)
  local s = GaPrint("\1TLGPC", ...)
  if type(s) ~= "string" then return end
  local out, i, n = {}, 1, 0
  while true do
    local j = string.find(s, "\31", i, true)
    n = n + 1
    if not j then out[n] = string.sub(s, i); break end
    out[n] = string.sub(s, i, j - 1)
    i = j + 1
  end
  return unpack(out, 1, n)
end
P.call = call
local function pack(...) return { n = select('#', ...), ... } end
local function show(t)
  local out = {}
  for i = 1, t.n do out[#out + 1] = tostring(t[i]) end
  return table.concat(out, " ")
end
function P.log(...) call("log", show(pack(...))) end

-- Pad buttons (DualShock 4 bits).
local UP, RIGHT, DOWN, LEFT = 0x10, 0x20, 0x40, 0x80
local CROSS, CIRCLE, SQUARE, TRIANGLE = 0x4000, 0x2000, 0x8000, 0x1000
local L1, R1 = 0x400, 0x800
local function has(v, b) return floor(v / b) % 2 == 1 end

-- The menu textures (the game's Options pages').
local UI = "game/textures/ui/gamemenu/"
local TEX = {
  bg = UI .. "sys_option_item_backguraund",
  cur = UI .. "sys_cursor",
  gage = UI .. "sys_option_item_cam_gage",
  left = UI .. "sys_option_item_current_left",
  right = UI .. "sys_option_item_current_right",
  up = UI .. "sys_option_item_cursor_up",
}

-- Layout (1920x1080 units, as the game's Options pages).
local L = {
  title_x = 113, title_y = 124, title_px = 34,
  row_x = 153, row_y = 207, row_dy = 81, rows = 5, row_px = 26,
  cur_x = 129,
  val_x0 = 480, val_x1 = 1010,
  help_x = 153, help_y = 596, help_px = 21, help_w = 850, help_dy = 26,
  bottom_y = 656, back_x = 153, reset_x = 320,
  panel_x0 = 50, panel_x1 = 1090, panel_y0 = 65, panel_y1 = 715,
}

-- While the frame-rate limit is the game's 30 fps, the frame-rate fixes
-- of the game's scripts (script_fixes.rs) keep every update one 30 Hz frame.
TLG_W30 = call("cap") == "game"

P.tex = P.tex or {}
P.cache = P.cache or {}
P.gen = P.gen or 0
P.n = 0

local function load_tex(key, name)
  if P.tex[key] == nil and name then
    P.tex[key] = false
    Load(name, function(t) P.tex[key] = t or false end)
  end
end

-- Text from the port's atlas: rect in texels, cached per atlas generation.
local function text(s, px, w)
  local k = s .. "|" .. px .. "|" .. (w or 400)
  local c = P.cache[k]
  if c and c.gen == P.gen then return c end
  local x, y, ww, hh, gen = call("text", s, tostring(px), tostring(w or 400))
  if not x then return nil end
  c = { x = tonumber(x), y = tonumber(y), w = tonumber(ww), h = tonumber(hh), gen = tonumber(gen) }
  if c.gen ~= P.gen then P.cache = {}; P.gen = c.gen end
  P.cache[k] = c
  return c
end

local function text_w(s, px, w)
  local c = text(s, px, w)
  return c and c.w or 0
end

-- s in lines at most w wide at px (at spaces; cached).
local function wrap(s, px, w)
  P.wraps = P.wraps or {}
  local k = s .. "|" .. px .. "|" .. w
  local l = P.wraps[k]
  if not l then
    l = { call("wrap", s, tostring(px), tostring(w)) }
    if #l == 0 then l = { s } end
    P.wraps[k] = l
  end
  return l
end

-- Draws text with its glyphs' left edge at x, centred on y; the game's
-- labels have a soft shadow below them and a slightly heavier stroke than
-- the atlas' (drawn twice, half a pixel apart).
local function draw_text(s, x, y, px, a, w, rgb)
  local t = P.tex.atlas
  if not t or s == "" then return 0 end
  local c = text(s, px, w)
  if not c then return 0 end
  local u0, v0, u1, v1 = c.x / P.aw, c.y / P.ah, (c.x + c.w) / P.aw, (c.y + c.h) / P.ah
  local top = y - c.h / 2
  -- (The atlas leaves px/8 texels, at least 2, before the glyphs.)
  x = x - math.max(2, floor(px / 8))
  a = a or 255
  GUI_SetTexture(t:GetName())
  GUI_DrawSubTextureEx(x + 1.5, top + 2, c.w, c.h, u0, v0, u1, v1, floor(a * 0.55), 0, 0, 0)
  local r, g, b = 255, 255, 255
  if rgb then r, g, b = rgb[1], rgb[2], rgb[3] end
  GUI_DrawSubTextureEx(x, top, c.w, c.h, u0, v0, u1, v1, a, r, g, b)
  if px >= 20 then GUI_DrawSubTextureEx(x + 0.5, top, c.w, c.h, u0, v0, u1, v1, floor(a * 0.45), r, g, b) end
  return c.w
end

local function draw_tex(key, x, y, w, h, a, u0, v0, u1, v1)
  local t = P.tex[key]
  if not t then return end
  GUI_SetTexture(t:GetName())
  GUI_DrawSubTextureEx(x, y, w, h, u0 or 0, v0 or 0, u1 or 1, v1 or 1, a or 255, 255, 255, 255)
end

-- ---------------------------------------------------------------- pages

local function split(s, sep)
  local out = {}
  if s == nil or s == "" then return out end
  local i = 1
  while true do
    local j = string.find(s, sep, i, true)
    if not j then out[#out + 1] = string.sub(s, i); break end
    out[#out + 1] = string.sub(s, i, j - 1)
    i = j + string.len(sep)
  end
  return out
end

-- Fetches a page: P.page = { id, title, rows = { {kind, key, label, value, opts, help, flags}... } }
local function fetch(id)
  local r = pack(call("page", id))
  if r.n < 2 then return nil end
  local pg = { id = id, title = r[1], rows = {} }
  local n = tonumber(r[2]) or 0
  for i = 0, n - 1 do
    local b = 3 + i * 7
    local row = { kind = r[b], key = r[b + 1], label = r[b + 2], value = r[b + 3], help = r[b + 5], flags = r[b + 6] or "" }
    local o = split(r[b + 4], "\t")
    if row.kind == "choice" then
      row.choices = {}
      for k = 1, #o - 1, 2 do row.choices[#row.choices + 1] = { v = o[k], l = o[k + 1] } end
    elseif row.kind == "slider" then
      row.min, row.max, row.step, row.unit = tonumber(o[1]) or 0, tonumber(o[2]) or 100, tonumber(o[3]) or 1, o[4] or ""
    elseif row.kind == "page" then
      row.to = o[1]
    elseif row.kind == "trophy" then
      row.uv = { tonumber(o[1]) or 0, tonumber(o[2]) or 0, tonumber(o[3]) or 0, tonumber(o[4]) or 0 }
    end
    pg.rows[#pg.rows + 1] = row
  end
  return pg
end

local function choice_index(row)
  for i, c in ipairs(row.choices) do
    if c.v == row.value then return i end
  end
  return nil
end

local function open_page(id, keep)
  local pg = fetch(id)
  if not pg then return end
  if not keep then
    P.sel, P.scroll = 1, 0
  end
  P.page = pg
  P.note = nil
  if P.sel > #pg.rows + 2 then P.sel = 1 end
end

-- The bottom row: Back, and Reset on the settings pages below the first.
local function bottom_items()
  if P.page and P.page.id ~= "main" and P.page.id ~= "achievements" then return { "back", "reset" } end
  return { "back" }
end

local function set_value(row, v)
  if v == nil or v == row.value then return end
  local r = call("set", row.key, v)
  if r == "ok" or r == "start" then
    row.value = v
    if r == "start" then P.note = "Applies at the next start." else P.note = nil end
    P.engine_gen = nil
  elseif r == "failed" then
    P.note = "Can't be changed right now."
  else
    P.log("set", row.key, v, "->", r)
  end
end

local function step_row(row, d)
  if row.kind == "choice" then
    local n = #row.choices
    if n == 0 then return end
    local i = choice_index(row) or 1
    i = i + d
    if i < 1 then i = 1 elseif i > n then i = n end
    set_value(row, row.choices[i].v)
  elseif row.kind == "slider" then
    local v = (tonumber(row.value) or row.min) + d * row.step
    if v < row.min then v = row.min elseif v > row.max then v = row.max end
    set_value(row, tostring(v))
  end
end

local function activate(row)
  if row.kind == "page" then
    P.stack = P.stack or {}
    P.stack[#P.stack + 1] = { id = P.page.id, sel = P.sel, scroll = P.scroll }
    open_page(row.to)
  elseif row.kind == "action" then
    local r = call("act", row.key)
    if r == "reload" then open_page(P.page.id, true) elseif r ~= "ok" then P.note = r end
  elseif row.kind == "choice" then
    step_row(row, 1)
  elseif row.kind == "bind" then
    if call("bind_start", row.key) == "ok" then
      P.capture = row
      P.note = nil
    end
  end
end

local function go_back()
  if P.confirm then P.confirm = nil; return end
  local s = P.stack and P.stack[#P.stack]
  if s then
    P.stack[#P.stack] = nil
    open_page(s.id)
    P.sel, P.scroll = s.sel, s.scroll
  else
    P.close()
  end
end

-- ---------------------------------------------------------------- open / close

function P.open_pages()
  P.open = true
  P.stack = {}
  P.confirm = nil
  open_page("main")
  call("opened")
  P.repeat_t, P.repeat_b = 0, 0
  P.fade = 0
end

function P.close()
  P.open = false
  P.page = nil
  P.capture = nil
  P.resume()
  call("closed")
end

-- F10 while playing: the game pauses as its pause button pauses it (the
-- engine's pause, the picture dimmed, the "GamePause" event), without its
-- menu; the pages close back into the game. Not while the pause menu shows
-- (its Options list has PC Settings) or where the game allows no pause.
function P.pause()
  if PauseMenuUI and (PauseMenuUI.isNowPauseMode or PauseMenuUI.pauseDelayCounter and PauseMenuUI.pauseDelayCounter >= 0) then return false end
  if TricoGame_IsInGame and TricoGame_IsInGame() and not TricoGame_IsGamePause() then
    if PauseMenuUI and (PauseMenuUI.isPauseDisableFlag or PauseMenuUI.isEndPauseRequest) then return false end
    if GameConfig and GameConfig.pauseMenuType == "IgnorePause" then return false end
    if MS and MS.IsGameOver and MS.IsGameOver() then return false end
    luaCommand("trico pause")
    PostEvent("GamePause", "system", "Start")
    local fc = FindObject("_fade_controller")
    if fc and fc.SystemValue == 1.0 then fc.SystemValue = 0.5; P.dimmed = true end
    P.paused = true
  end
  return true
end

function P.resume()
  if not P.paused then return end
  P.paused = false
  local fc = FindObject("_fade_controller")
  if P.dimmed and fc and fc.SystemValue == 0.5 then fc.SystemValue = 1.0 end
  P.dimmed = false
  if TricoGame_IsGamePause() then luaCommand("trico resume") end
  PostEvent("GamePause", "system", "End")
end

-- ---------------------------------------------------------------- input

local REPEAT_DELAY, REPEAT_EVERY = 0.4, 0.09

-- Directions pressed this frame, with held repeats (pad, keys, stick).
local function directions(pressed, held, sx, sy, dt)
  local dirs = 0
  local hold = 0
  if has(held, UP) or sy < 0 then hold = UP
  elseif has(held, DOWN) or sy > 0 then hold = DOWN
  elseif has(held, LEFT) or sx < 0 then hold = LEFT
  elseif has(held, RIGHT) or sx > 0 then hold = RIGHT end
  for _, b in ipairs({ UP, DOWN, LEFT, RIGHT }) do
    if has(pressed, b) then dirs = dirs + b end
  end
  if hold ~= 0 then
    if P.repeat_b ~= hold then
      P.repeat_b, P.repeat_t = hold, -REPEAT_DELAY
      if dirs == 0 and (sx ~= 0 or sy ~= 0) then dirs = hold end
    else
      P.repeat_t = P.repeat_t + dt
      if P.repeat_t >= REPEAT_EVERY then
        P.repeat_t = 0
        if not has(dirs, hold) then dirs = dirs + hold end
      end
    end
  else
    P.repeat_b = 0
  end
  return dirs
end

-- The rows' positions on screen (visible ones): index -> y.
local function row_y(i)
  return L.row_y + (i - 1 - P.scroll) * L.row_dy
end

local function visible(i)
  return i > P.scroll and i <= P.scroll + L.rows
end

local function keep_visible()
  local n = #P.page.rows
  if P.sel <= n then
    if P.sel <= P.scroll then P.scroll = P.sel - 1 end
    if P.sel > P.scroll + L.rows then P.scroll = P.sel - L.rows end
  end
  if P.scroll > n - L.rows then P.scroll = math.max(0, n - L.rows) end
end

-- Value slots of a choice row: { {x0, x1, centre}... } or nil (compact).
local function slots(row)
  local n = #row.choices
  if n == 0 then return nil end
  local total = 0
  for _, c in ipairs(row.choices) do total = total + text_w(c.l, L.row_px) + 70 end
  if total > (L.val_x1 - L.val_x0) then return nil end
  local w = (L.val_x1 - L.val_x0) / n
  local out = {}
  for i = 1, n do
    local x0 = L.val_x0 + (i - 1) * w
    out[i] = { x0 = x0, x1 = x0 + w, c = x0 + w / 2 }
  end
  return out
end

local GAGE_X0, GAGE_X1 = 550, 956

local function pointer_input(mx, my, moved, mp, wheel)
  local rows = P.page.rows
  local n = #rows
  local hit = nil
  if mx >= L.panel_x0 and mx <= L.panel_x1 then
    for i = P.scroll + 1, math.min(n, P.scroll + L.rows) do
      local y = row_y(i)
      if my >= y - L.row_dy / 2 and my < y + L.row_dy / 2 then hit = i end
    end
    if my >= L.bottom_y - 30 and my < L.bottom_y + 30 then
      local items = bottom_items()
      if mx < L.reset_x - 20 or #items == 1 then hit = n + 1 else hit = n + 2 end
    end
  end
  if moved and hit then P.sel = hit end
  if wheel ~= 0 then
    local d = wheel > 0 and -1 or 1
    P.scroll = P.scroll + d
    keep_visible()
    if P.scroll < 0 then P.scroll = 0 end
  end
  if has(mp, 2) then go_back(); return end
  if not has(mp, 1) or not hit then return end
  P.sel = hit
  if hit > n then
    if hit == n + 1 then go_back() else P.confirm = { yes = false } end
    return
  end
  local row = rows[hit]
  if row.kind == "choice" then
    local sl = slots(row)
    if sl then
      for i, s in ipairs(sl) do
        if mx >= s.x0 and mx < s.x1 then set_value(row, row.choices[i].v) end
      end
    elseif mx >= L.val_x0 then
      step_row(row, mx < (L.val_x0 + L.val_x1) / 2 and -1 or 1)
    else
      step_row(row, 1)
    end
  elseif row.kind == "slider" then
    if mx >= GAGE_X0 - 20 and mx <= GAGE_X1 + 20 then
      local f = (mx - GAGE_X0) / (GAGE_X1 - GAGE_X0)
      if f < 0 then f = 0 elseif f > 1 then f = 1 end
      local v = row.min + f * (row.max - row.min)
      v = floor(v / row.step + 0.5) * row.step
      set_value(row, tostring(v))
    end
  else
    activate(row)
  end
end

local function confirm_input(dirs, pressed, mx, my, mp, moved)
  local c = P.confirm
  if has(dirs, LEFT) or has(dirs, RIGHT) then c.yes = not c.yes end
  local yes_x, no_x, y = 700, 860, L.bottom_y
  if moved and my > y - 30 and my < y + 30 then
    if mx >= yes_x - 20 and mx < no_x - 20 then c.yes = true elseif mx >= no_x - 20 and mx < no_x + 120 then c.yes = false end
  end
  local click = has(mp, 1) and my > y - 30 and my < y + 30 and mx >= yes_x - 20 and mx < no_x + 120
  if has(pressed, CROSS) or click then
    if c.yes then call("reset", P.page.id); open_page(P.page.id, true) end
    P.confirm = nil
  elseif has(pressed, CIRCLE) or has(mp, 2) then
    P.confirm = nil
  end
end

local function capture_input()
  local r = call("bind_poll")
  if r == "wait" then return end
  local row = P.capture
  P.capture = nil
  if r == "done" or r == "cleared" or r == "cancel" then
    P.note = nil
  else
    P.note = r
  end
  if r ~= "cancel" and P.page then open_page(P.page.id, true) end
end

local function page_input(dt)
  local pressed, held, sx, sy, mx, my, moved, mp, mh, wheel = call("input")
  if P.capture then
    capture_input()
    return
  end
  pressed, held = tonumber(pressed) or 0, tonumber(held) or 0
  sx, sy = tonumber(sx) or 0, tonumber(sy) or 0
  mx, my = tonumber(mx) or 960, tonumber(my) or 540
  mp, wheel = tonumber(mp) or 0, tonumber(wheel) or 0
  moved = moved == "1"
  P.mx, P.my = mx, my
  if moved then P.pointer_t = 3 end
  local dirs = directions(pressed, held, sx, sy, dt)
  if P.confirm then confirm_input(dirs, pressed, mx, my, mp, moved); return end
  local rows = P.page.rows
  local n = #rows
  local items = bottom_items()
  local last = n + #items
  if has(dirs, UP) then
    if P.sel > n then P.sel = n else P.sel = math.max(1, P.sel - 1) end
  elseif has(dirs, DOWN) then
    if P.sel < n then P.sel = P.sel + 1 elseif P.sel == n then P.sel = n + 1 end
  end
  if P.sel <= n then
    local row = rows[P.sel]
    if has(dirs, LEFT) then step_row(row, -1) elseif has(dirs, RIGHT) then step_row(row, 1) end
  else
    if has(dirs, LEFT) and P.sel > n + 1 then P.sel = P.sel - 1 elseif has(dirs, RIGHT) and P.sel < last then P.sel = P.sel + 1 end
  end
  keep_visible()
  if has(pressed, CIRCLE) then go_back(); return end
  if has(pressed, CROSS) then
    if P.sel > n then
      if P.sel == n + 1 then go_back() else P.confirm = { yes = false } end
    else
      activate(rows[P.sel])
    end
    return
  end
  pointer_input(mx, my, moved, mp, wheel)
end

-- ---------------------------------------------------------------- drawing

-- (The game's cursor and arrow glyphs fill about 0.28 of their textures.)
local function draw_cursor(x, y, a)
  draw_tex("cur", x - 38, y - 38, 76, 76, a)
end

-- The game's brackets around the current choice.
local function draw_brackets(x, w, y, a)
  draw_tex("left", x - 72, y - 52, 104, 104, a)
  draw_tex("right", x + w - 32, y - 52, 104, 104, a)
end

local function draw_choice(row, y, sel)
  local sl = slots(row)
  local cur = choice_index(row)
  if sl then
    for i, c in ipairs(row.choices) do
      local w = text_w(c.l, L.row_px)
      local x = sl[i].c - w / 2
      local on = (i == cur)
      draw_text(c.l, x, y, L.row_px, on and 255 or 110)
      if on then draw_brackets(x, w, y, 255) end
    end
  else
    local label = cur and row.choices[cur].l or row.value
    local w = text_w(label, L.row_px)
    local cx = (L.val_x0 + L.val_x1) / 2
    local x = cx - w / 2
    draw_text(label, x, y, L.row_px, 255)
    draw_brackets(x, w, y, 255)
    -- (More choices to either side: the game's cursor, pointing there.)
    local a = sel and 230 or 120
    if cur == nil or cur > 1 then draw_tex("cur", L.val_x0 - 10, y - 32, 64, 64, a, 1, 0, 0, 1) end
    if cur == nil or cur < #row.choices then draw_tex("cur", L.val_x1 - 54, y - 32, 64, 64, a) end
  end
end

local function draw_slider(row, y)
  draw_tex("gage", GAGE_X0, y - 26, GAGE_X1 - GAGE_X0, 50, 255)
  local v = tonumber(row.value) or row.min
  local f = (v - row.min) / math.max(1, row.max - row.min)
  local x = GAGE_X0 + f * (GAGE_X1 - GAGE_X0)
  draw_tex("up", x - 32, y + 2, 64, 64, 255)
  draw_text(tostring(v) .. row.unit, GAGE_X1 + 34, y - 4, L.row_px - 2, 200)
end

local function draw_page()
  local pg = P.page
  local a = floor(255 * math.min(1, P.fade))
  draw_tex("bg", 0, 0, 1920, 1080, a)
  draw_text(pg.title, L.title_x, L.title_y, L.title_px, a)
  local n = #pg.rows
  for i = P.scroll + 1, math.min(n, P.scroll + L.rows) do
    local row = pg.rows[i]
    local y = row_y(i)
    local sel = (P.sel == i) and not P.confirm
    local label = row.label
    if row.kind == "trophy" then
      -- A trophy: its icon, name and when it unlocked (dimmed while locked).
      local on = string.find(row.flags, "u", 1, true) ~= nil
      local t = P.tex.trophies
      if t and row.uv then
        local g = on and 255 or 120
        GUI_SetTexture(t:GetName())
        GUI_DrawSubTextureEx(L.row_x + 4, y - 32, 64, 64, row.uv[1], row.uv[2], row.uv[3], row.uv[4], floor(a * (on and 1 or 0.6)), g, g, g)
      end
      local px = L.row_px
      if text_w(label, px) > 560 then px = px - 4 end
      draw_text(label, L.row_x + 88, y, px, floor(a * (on and 1 or 0.6)))
      local w = text_w(row.value, L.row_px - 2)
      draw_text(row.value, L.val_x1 - w, y, L.row_px - 2, floor(a * (on and 0.85 or 0.5)))
    else
      draw_text(label, L.row_x, y, L.row_px, a)
    end
    if sel then draw_cursor(L.cur_x, y, a) end
    if row.kind == "choice" then draw_choice(row, y, sel)
    elseif row.kind == "slider" then draw_slider(row, y)
    elseif row.kind == "action" and row.value and row.value ~= "" then
      local w = text_w(row.value, L.row_px - 2)
      draw_text(row.value, L.val_x1 - w, y, L.row_px - 2, floor(a * 0.85))
    elseif row.kind == "bind" then
      local v = row.value
      if P.capture == row then v = "..." end
      local w = text_w(v, L.row_px)
      local x = (L.val_x0 + L.val_x1) / 2 - w / 2
      draw_text(v, x, y, L.row_px, (v == "-") and 110 or 255)
      if sel then draw_brackets(x, w, y, 255) end
    end
  end
  -- (More rows above or below: the game's scroll arrows.)
  if P.scroll > 0 then draw_tex("up", 1012, 140, 64, 64, a) end
  if P.scroll + L.rows < n then draw_tex("up", 1012, row_y(P.scroll + L.rows) + 22, 64, 64, a, 0, 1, 1, 0) end
  -- The selected row's help, or a note.
  local help = P.note
  if P.capture then
    help = "Press a key or mouse button for " .. P.capture.label .. "   (Esc: cancel, Backspace: clear)"
  end
  if not help and P.sel <= n and pg.rows[P.sel] then
    local row = pg.rows[P.sel]
    help = row.help
    if string.find(row.flags, "p", 1, true) then help = "Applies at the next start." end
  end
  if help and help ~= "" then
    -- (A long one on up to three lines around the help line, clear of the
    -- scroll arrow.)
    local lines = wrap(help, L.help_px, L.help_w)
    local nl = math.min(#lines, 3)
    for i = 1, nl do
      draw_text(lines[i], L.help_x, L.help_y + floor((i - (nl + 1) / 2) * L.help_dy), L.help_px, floor(a * 0.75))
    end
  end
  -- Back and Reset, or the reset question.
  if P.confirm then
    draw_text("Restore this page's defaults?", L.back_x, L.bottom_y, L.row_px, a)
    local yes_x, no_x = 700, 860
    draw_text("Yes", yes_x, L.bottom_y, L.row_px, a)
    draw_text("No", no_x, L.bottom_y, L.row_px, a)
    draw_cursor((P.confirm.yes and yes_x or no_x) - 24, L.bottom_y, a)
  else
    local items = bottom_items()
    draw_text("Back", L.back_x, L.bottom_y, L.row_px, a)
    if #items > 1 then draw_text("Reset", L.reset_x, L.bottom_y, L.row_px, a) end
    if P.sel == n + 1 then draw_cursor(L.cur_x, L.bottom_y, a) end
    if P.sel == n + 2 then draw_cursor(L.reset_x - 24, L.bottom_y, a) end
  end
end

local function pointer_at(x, y, a)
  if not P.tex.atlas then return end
  GUI_SetTexture(P.tex.atlas:GetName())
  for _, part in ipairs({ { "\1pointer_rim", 0, 0.8 }, { "\1pointer", 255, 1 } }) do
    local c = text(part[1], 30, 400)
    if c then
      GUI_DrawSubTextureEx(x - 1, y - 1, c.w, c.h, c.x / P.aw, c.y / P.ah, (c.x + c.w) / P.aw, (c.y + c.h) / P.ah, floor(a * part[3]), part[2], part[2], part[2])
    end
  end
end

local function draw_pointer()
  if not P.pointer_t or P.pointer_t <= 0 or not P.mx then return end
  pointer_at(P.mx, P.my, floor(255 * math.min(1, P.pointer_t)))
end

-- The pointer in the game's own menus (menu_pointer.rs moves it and
-- selects what it points at).
local function draw_menu_pointer()
  local x, y, shown = call("menu_ptr")
  local s = tonumber(shown)
  if not s then return end
  pointer_at(tonumber(x) or 960, tonumber(y) or 540, floor(255 * math.min(1, s)))
end

-- ---------------------------------------------------------------- engine settings

-- The port's overrides of the engine's objects, kept in force each frame:
-- each remembers the game's own value, put back when the override goes.
P.ov = P.ov or {}

local function near(a, b)
  local x, y = tonumber(a), tonumber(b)
  if x and y then return math.abs(x - y) <= 1e-4 * math.max(1, math.abs(x)) end
  return tostring(a) == tostring(b)
end

local function has_flag(s, name)
  s = "|" .. tostring(s or "") .. "|"
  return string.find(s, "|" .. name .. "|", 1, true) ~= nil
end

local function enforce(o)
  local obj = FindObject(o.obj)
  if not obj then return end
  local ok, err = pcall(function()
    if o.mode == "flag" then
      local cur = has_flag(obj[o.field], o.flag)
      if o.written == nil or cur ~= o.written then o.game = cur end
      if cur ~= o.on then
        obj[o.field] = (o.on and "|" or "~") .. o.flag
      end
      o.written = o.on
    else
      local cur = obj[o.field]
      if o.written == nil or not near(cur, o.written) then o.game = cur end
      local want = o.value
      if o.mode == "mul" then want = (tonumber(o.game) or 0) * (tonumber(o.value) or 1) end
      if not near(cur, want) then
        obj[o.field] = (tonumber(want) and o.mode ~= "str") and tonumber(want) or want
      end
      o.written = obj[o.field]
    end
  end)
  if not ok and not o.err then o.err = true; P.log("override", o.obj, o.field, "failed:", tostring(err)) end
end

local function restore(o)
  local obj = FindObject(o.obj)
  if not obj or o.game == nil then return end
  pcall(function()
    if o.mode == "flag" then
      obj[o.field] = (o.game and "|" or "~") .. o.flag
    else
      obj[o.field] = o.game
    end
  end)
end

local function engine_tick()
  local gen = call("engine_gen")
  if gen ~= P.engine_gen then
    P.engine_gen = gen
    -- The Extras page: globals the game's changed scripts test
    -- (script_fixes.rs EXTRAS), true while a setting is on.
    local x = pack(call("extras"))
    for i = 1, x.n - 1, 2 do
      if x[i] and x[i] ~= "" then _G[x[i]] = x[i + 1] == "1" end
    end
    local r = pack(call("engine"))
    local new = {}
    local n = tonumber(r[1]) or 0
    for i = 0, n - 1 do
      local b = 2 + i * 4
      local o = { obj = r[b], field = r[b + 1], mode = r[b + 2], value = r[b + 3] }
      if o.mode == "flag" then
        local name, st = string.match(o.value, "^(.-)=(.*)$")
        o.flag, o.on = name, st == "on"
      end
      if o.mode == "set" and not tonumber(o.value) then o.mode = "str" end
      local key = o.obj .. "." .. o.field .. "." .. (o.flag or "")
      local prev = P.ov[key]
      if prev then o.game, o.written = prev.game, prev.written end
      new[key] = o
    end
    for key, o in pairs(P.ov) do
      if not new[key] then restore(o) end
    end
    P.ov = new
  end
  for _, o in pairs(P.ov) do enforce(o) end
end

-- ---------------------------------------------------------------- button prompts

-- Each of the game's prompts (guides) gets a picture of its own: the port
-- draws it for the device in use (pc_prompts.rs: the keys of that prompt
-- on a keyboard, the Xbox pad, or the game's DualShock). The pictures are
-- textures of the port's archive, loaded here ahead of the guides; when
-- the game makes a guide, its picture is the slot for the guide's buttons.
P.slot_tex = P.slot_tex or {}

local function load_slots()
  if P.slots_loading then return end
  P.slots_loading = true
  local names = pack(call("guide_slots"))
  for i = 1, names.n do
    local n = names[i]
    if n and n ~= "" then
      Load(n, function(t) if t then P.slot_tex[n] = t end end)
    end
  end
end

-- The guide data made with this name: its buttons, as "Circle+StickBaseL".
local function guide_buttons(name)
  local list = GuideSystem and GuideSystem.data_list
  if not list then return "" end
  for _, d in ipairs(list) do
    if d.sysParam and d.sysParam.DbgName == name then
      local ids = {}
      local ps = d.effect and d.effect.Param
      if ps then
        for _, e in ipairs(ps) do
          if e.uvID and e.uvID ~= "" then ids[#ids + 1] = e.uvID end
        end
      end
      return table.concat(ids, "+")
    end
  end
  return ""
end

local function wrap_guides()
  if P.guides_wrapped or not Guide_CreateGuide then return end
  P.guides_wrapped = true
  local orig = Guide_CreateGuide
  -- (Its 10th argument is the picture's texture name; the rest as given.)
  Guide_CreateGuide = function(name, id, norma, group, prio, mode, lines, title, frame, pad, effect, ...)
    local ok, slot = pcall(function() return call("guide_tex", guide_buttons(name)) end)
    if ok and slot and slot ~= "" then
      local tex = P.slot_tex[slot]
      if tex then pad = tex:GetName() else call("guide_tex", "") end
    end
    return orig(name, id, norma, group, prio, mode, lines, title, frame, pad, effect, ...)
  end
end

wrap_guides()

-- ---------------------------------------------------------------- console (development)

local function console()
  local codes = pack(call("lua"))
  for i = 1, codes.n do
    local code = codes[i]
    if code and code ~= "" then
      local f, err = loadstring(code)
      if not f then
        call("log", "lua error: " .. tostring(err))
      else
        local echo = #code > 200 and (string.sub(code, 1, 200) .. "...") or code
        call("log", "lua> " .. echo .. " => " .. show(pack(pcall(f))))
      end
    end
  end
end

-- ---------------------------------------------------------------- achievements

-- An unlocked trophy's pop-up: top right of the picture's 16:9 area in the
-- game's guide-box look, one at a time, about 5 s each (achievements.rs).
P.toasts = P.toasts or {}

local function trophy_tick(dt)
  if not P.ttex then
    local i, f, fw, fh = call("trophy_tex")
    if not i then return end
    P.ttex = { icons = i, frame = f, w = tonumber(fw) or 544, h = tonumber(fh) or 120 }
  end
  load_tex("trophies", P.ttex.icons)
  load_tex("tframe", P.ttex.frame)
  local r = pack(call("trophy_toast"))
  if r.n >= 6 and r[1] then
    P.toasts[#P.toasts + 1] = { name = r[1], above = r[2], uv = { tonumber(r[3]) or 0, tonumber(r[4]) or 0, tonumber(r[5]) or 0, tonumber(r[6]) or 0 }, t = 0 }
  end
  local q = P.toasts[1]
  if not q or not P.tex.tframe or not P.tex.trophies or not P.tex.atlas then return end
  q.t = q.t + dt
  if q.t > 5.6 then
    table.remove(P.toasts, 1)
    return
  end
  local a = math.min(1, q.t / 0.35, math.max(0, (5.6 - q.t) / 0.6))
  local A = floor(255 * a)
  local w, h = P.ttex.w, P.ttex.h
  local x, y = 1920 - 64 - w, 52 - (1 - math.min(1, q.t / 0.35)) * 24
  draw_tex("tframe", x, y, w, h, A)
  GUI_SetTexture(P.tex.trophies:GetName())
  GUI_DrawSubTextureEx(x + 16, y + 16, h - 32, h - 32, q.uv[1], q.uv[2], q.uv[3], q.uv[4], A, 255, 255, 255)
  local px = 26
  while px > 18 and text_w(q.name, px) > w - h - 24 do px = px - 2 end
  draw_text(q.above, x + h + 4, y + h * 0.33, 20, floor(A * 0.75))
  draw_text(q.name, x + h + 4, y + h * 0.66, px, A)
end

-- ---------------------------------------------------------------- tick

P.Tick = function()
  P.n = P.n + 1
  -- (Real time: while the game is paused its own step is 0.)
  local now = tonumber(call("now")) or 0
  local dt = P.last_now and (now - P.last_now) or (1 / 30)
  P.last_now = now
  if dt < 0 then dt = 0 elseif dt > 0.25 then dt = 0.25 end
  if not P.hello then
    P.hello = call("hello")
    if not P.hello then return end
    local n, w, h = call("atlas")
    P.atlas_name, P.aw, P.ah = n, tonumber(w), tonumber(h)
  end
  console()
  TLG_W30 = call("cap") == "game"
  engine_tick()
  load_slots()
  wrap_guides()
  load_tex("atlas", P.atlas_name)
  trophy_tick(dt)
  if P.dbg then pcall(P.dbg, dt) end
  if not P.open then
    -- The Options list's PC Settings row: its label, fading with the list.
    local ix, iy, label = call("item")
    P.item_a = P.item_a or 0
    if ix then
      P.item_a = math.min(1, P.item_a + dt * 6)
      P.item_x, P.item_y, P.item_label = tonumber(ix), tonumber(iy), label
    else
      P.item_a = math.max(0, P.item_a - dt * 6)
    end
    if P.item_a > 0 and P.item_x and P.tex.atlas then
      draw_text(P.item_label, P.item_x, P.item_y, L.row_px, floor(255 * P.item_a))
    end
    draw_menu_pointer()
    local req = call("poll")
    if req == "open" then
      if call("origin") == "options" or P.pause() then P.open_pages() else call("refused") end
    end
    return
  end
  for k, name in pairs(TEX) do load_tex(k, name) end
  P.fade = math.min(1, (P.fade or 0) + dt * 6)
  if P.pointer_t then P.pointer_t = P.pointer_t - dt end
  page_input(dt)
  if not P.open or not P.page then return end
  draw_page()
  draw_pointer()
end

-- (Once: a reloaded script - development - only replaces P.Tick.)
if not P.wrapped then
  P.wrapped = true
  local orig = Tick
  Tick = function(dt, edt)
    local r = orig(dt, edt)
    local ok, err = pcall(P.Tick, dt)
    if not ok and not P.err then P.err = true; call("log", "tick error: " .. tostring(err)) end
    return r
  end
end
