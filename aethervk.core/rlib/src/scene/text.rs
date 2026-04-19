use hashbrown::HashMap;
use alloc::vec::Vec;
use alloc::vec;
use ab_glyph::{FontRef, Font, PxScale, ScaleFont, point};

pub struct FontAtlas {
  pub image_data: Vec<u8>,
  pub width: u32,
  pub height: u32,
  pub glyphs: HashMap<char, GlyphInfo>,
  pub line_height: f32,
  pub ascent: f32,
  pub descent: f32,
  pub line_gap: f32,
}

#[derive(Clone, Copy)]
pub struct GlyphInfo {
  pub uv_min: [f32; 2],
  pub uv_max: [f32; 2],
  pub size: [f32; 2],
  pub offset: [f32; 2],
  pub advance: f32,
}

pub fn create_ascii_atlas(font_data: &[u8], scale_pt: f32) -> Option<FontAtlas> {
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

  let chars = (32..=126).map(|c| c as u8 as char).chain(core::iter::once('█'));

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
        outline: outlined, // just pass the outlined glyph directly
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
      // Space or empty glyph
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
        image_data[idx] = val; // Store alpha in red channel
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
  })
}
