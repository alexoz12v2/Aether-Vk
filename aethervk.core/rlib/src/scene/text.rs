use hashbrown::HashMap;
use alloc::vec::Vec;
use core::hash::Hasher;
use ab_glyph::{FontRef, Font, PxScale, ScaleFont, point};
use aethervk_oshal_rlib::hash::FnvHasher;
use aethervk_oshal_rlib::os;
use aethervk_oshal_rlib::os::fs::Path;
use crate::types::{IoError, IoResult};

pub struct FontAtlas {
  pub image_data: Vec<u8>,
  pub width: u32,
  pub height: u32,
  pub glyphs: HashMap<char, GlyphInfo>,
  pub line_height: f32,
  pub ascent: f32,
  pub descent: f32,
  pub line_gap: f32,
  /// not part of metadata for hash cause other values are derived from this
  /// took because it's needed for rendering code
  pub scale: PxScale,
}

#[derive(Clone, Copy)]
pub struct GlyphInfo {
  pub uv_min: [f32; 2],
  pub uv_max: [f32; 2],
  pub size: [f32; 2],
  pub offset: [f32; 2],
  pub advance: f32,
}

impl GlyphInfo {
  pub fn scaled_advance(&self, desired_points: f32, atlas_scale: PxScale) -> f32 {
    let scale_factor = desired_points / atlas_scale.x;
    self.advance * scale_factor
  }

  pub fn uv_bounds(&self) -> [f32; 4] {
    [
      self.uv_min[0],
      self.uv_min[1],
      self.uv_max[0],
      self.uv_max[1],
    ]
  }

  /// Get center position of glyph from cursor position (bottom left of glyph bounding box)
  pub fn screen_position(
    &self,
    cursor_position: [f32; 2],
    screen_extent: [f32; 2],
    desired_points: f32,
    atlas_scale: PxScale,
  ) -> [f32; 2] {
    let (scale_x, scale_y) = Self::screen_scale(screen_extent, desired_points, atlas_scale);
    [
      cursor_position[0] + self.offset[0] * scale_x,
      cursor_position[1] + self.offset[1] * scale_y,
    ]
  }

  /// duplicated computation with screen position. Shouldn't be such an overhead. Maintained for clarity
  pub fn screen_size(
    &self,
    screen_extent: [f32; 2],
    desired_points: f32,
    atlas_scale: PxScale,
  ) -> [f32; 2] {
    let (scale_x, scale_y) = Self::screen_scale(screen_extent, desired_points, atlas_scale);
    [self.size[0] * scale_x, self.size[1] * scale_y]
  }

  fn screen_scale(
    screen_extent: [f32; 2],
    desired_points: f32,
    atlas_scale: PxScale,
  ) -> (f32, f32) {
    let scale_x = (desired_points / atlas_scale.x) * 2.0 / screen_extent[0];
    let scale_y = (desired_points / atlas_scale.y) * 2.0 / screen_extent[1];
    (scale_x, scale_y)
  }
}

impl FontAtlas {
  pub fn scaled_height(&self, desired_points: f32) -> f32 {
    let scale_factor = desired_points / self.scale.y;
    scale_factor * self.line_height
  }

  pub fn hash_metadata(&self) -> u64 {
    let mut hasher = FnvHasher::new();
    hasher.write(self.width.to_be_bytes().as_slice());
    hasher.write(self.height.to_be_bytes().as_slice());
    hasher.write(self.ascent.to_be_bytes().as_slice());
    hasher.write(self.descent.to_be_bytes().as_slice());
    hasher.write(self.line_gap.to_be_bytes().as_slice());
    hasher.finish()
  }

  pub fn from_path<P: AsRef<Path>>(path: P, scale_pt: f32) -> IoResult<Self> {
    let font_data = os::fs::read(path).map_err(|e| IoError::from(e))?;
    Self::from_slice(&font_data, scale_pt)
      .ok_or_else(|| IoError::Specific("Failed to parse font data or create atlas"))
  }

  pub fn from_slice(font_data: &[u8], scale_pt: f32) -> Option<Self> {
    let font = FontRef::try_from_slice(font_data).ok()?;
    let scale = PxScale::from(scale_pt);
    let scaled_font = font.as_scaled(scale);
    let ascent = scaled_font.ascent();
    let descent = scaled_font.descent();
    let line_gap = scaled_font.line_gap();

    let mut glyphs = HashMap::new();
    let mut current_x = 0;
    let mut current_y = 0;
    let mut row_height = 0;

    let atlas_width = 1024;

    struct PlacedGlyph {
      c: char,
      outline: ab_glyph::OutlinedGlyph,
      bb: ab_glyph::Rect,
      x: u32,
      y: u32,
      advance: f32,
    }

    let mut placed = Vec::new();
    let chars = (32..=126)
      .map(|c| c as u8 as char)
      .chain(core::iter::once('█'));

    for c in chars {
      let glyph = font.glyph_id(c).with_scale(scale);
      let advance = scaled_font.h_advance(glyph.id);

      if let Some(outlined) = font.outline_glyph(glyph.clone()) {
        let bb = outlined.px_bounds();
        let gw = (bb.max.x - bb.min.x).ceil() as u32 + 2;
        let gh = (bb.max.y - bb.min.y).ceil() as u32 + 2;

        if current_x + gw >= atlas_width {
          current_x = 0;
          current_y += row_height + 2;
          row_height = 0;
        }

        placed.push(PlacedGlyph {
          c,
          outline: outlined,
          bb,
          x: current_x + 1,
          y: current_y + 1,
          advance,
        });

        current_x += gw;
        if gh > row_height {
          row_height = gh;
        }
      } else {
        glyphs.insert(
          c,
          GlyphInfo {
            uv_min: [0.0, 0.0],
            uv_max: [0.0, 0.0],
            size: [0.0, 0.0],
            offset: [0.0, 0.0],
            advance,
          },
        );
      }
    }

    let atlas_height = current_y + row_height + 2;
    let atlas_height = (atlas_height + 3) & !3;

    let mut image_data = alloc::vec![0u8; (atlas_width * atlas_height) as usize];

    for p in placed {
      let gw = (p.bb.max.x - p.bb.min.x).ceil() as u32;
      let gh = (p.bb.max.y - p.bb.min.y).ceil() as u32;

      p.outline.draw(|x, y, v| {
        let px = p.x + x;
        let py = p.y + y;
        if px < atlas_width && py < atlas_height {
          let idx = (py * atlas_width + px) as usize;
          let val = (v * 255.0) as u8;
          image_data[idx] = val;
        }
      });

      glyphs.insert(
        p.c,
        GlyphInfo {
          uv_min: [
            p.x as f32 / atlas_width as f32,
            p.y as f32 / atlas_height as f32,
          ],
          uv_max: [
            (p.x + gw) as f32 / atlas_width as f32,
            (p.y + gh) as f32 / atlas_height as f32,
          ],
          size: [gw as f32, gh as f32],
          offset: [p.bb.min.x, p.bb.min.y],
          advance: p.advance,
        },
      );
    }

    Some(FontAtlas {
      image_data,
      width: atlas_width,
      height: atlas_height,
      glyphs,
      line_height: ascent - descent + line_gap,
      ascent,
      descent,
      line_gap,
      scale,
    })
  }
}
