-- Image widths by file name, so reports/REPORT.md stays plain Markdown for GitHub.
-- Used by `make report` and `make report-pdf`. Widths are % of the text width; images inside
-- a table (two side by side) all get IN_TABLE, which reports/report.css maps to the cell width.

local WIDTHS = {
  architecture = "100%",
  eda_missingness_by_site = "100%",
  model_comparison = "100%",
  model_loso_by_site = "80%",
}
local SCREENSHOT = "55%" -- body screenshots; the appendix ones sit in two-column tables
local IN_TABLE = "46%" -- two figures side by side in a two-column table

local function width_for(src)
  local name = src:match("([^/]+)%.%w+$") or src
  if src:find("screenshots/", 1, true) then
    return SCREENSHOT
  end
  return WIDTHS[name]
end

local function set_width(img, width)
  if width and not img.attributes.width then
    img.attributes.width = width
  end
  return img
end

-- Two passes: tables first, so their images keep the side-by-side width.
return {
  {
    Table = function(tbl)
      return tbl:walk({
        Image = function(img)
          return set_width(img, IN_TABLE)
        end,
      })
    end,
  },
  {
    Image = function(img)
      if img.src:find("screenshots/", 1, true) and not img.attributes.width then
        img.classes:insert("screenshot") -- a body screenshot: reports/report.css caps its height
      end
      return set_width(img, width_for(img.src))
    end,
  },
}
