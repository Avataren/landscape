//! Procedural grass blade mesh generation.
//!
//! Generates 8 variants of grass blades with different bend angles, tip curls, and asymmetries.
//! Each blade is a simple 2-sided quad mesh suitable for GPU instancing.

use bevy::prelude::*;

/// Configuration for grass blade generation.
#[derive(Clone, Debug)]
pub struct GrassBladeMeshConfig {
    /// Height of the blade in local units (Y axis).
    pub blade_height: f32,
    /// Width at the base of the blade.
    pub blade_width: f32,
    /// Number of segments along the blade height (more = smoother curves).
    pub segments: u32,
}

impl Default for GrassBladeMeshConfig {
    fn default() -> Self {
        Self {
            blade_height: 1.0,
            blade_width: 0.2,
            segments: 4,
        }
    }
}

/// Variant parameters for grass blade geometry.
#[derive(Clone, Copy, Debug)]
pub struct GrassBladeVariant {
    /// Forward bend (0..1): 0 = straight, 1 = fully bent forward (45°)
    pub bend_forward: f32,
    /// Tip curl (0..1): 0 = no curl, 1 = 90° curl at tip
    pub tip_curl: f32,
    /// Left/right asymmetry (0..1): 0 = symmetric, 1 = pushed to one side
    pub asymmetry: f32,
}

impl GrassBladeVariant {
    /// Create the 8 default grass blade variants.
    pub fn create_variants() -> [GrassBladeVariant; 8] {
        [
            // Variant 0: mostly straight
            GrassBladeVariant {
                bend_forward: 0.0,
                tip_curl: 0.05,
                asymmetry: 0.0,
            },
            // Variant 1: slight forward bend
            GrassBladeVariant {
                bend_forward: 0.1,
                tip_curl: 0.1,
                asymmetry: 0.1,
            },
            // Variant 2: moderate forward bend
            GrassBladeVariant {
                bend_forward: 0.25,
                tip_curl: 0.15,
                asymmetry: -0.15,
            },
            // Variant 3: strong forward bend
            GrassBladeVariant {
                bend_forward: 0.4,
                tip_curl: 0.2,
                asymmetry: 0.2,
            },
            // Variant 4: mild curl, asymmetric
            GrassBladeVariant {
                bend_forward: 0.05,
                tip_curl: 0.3,
                asymmetry: -0.1,
            },
            // Variant 5: curled tip, slight bend
            GrassBladeVariant {
                bend_forward: 0.15,
                tip_curl: 0.35,
                asymmetry: 0.25,
            },
            // Variant 6: heavily bent and curled
            GrassBladeVariant {
                bend_forward: 0.35,
                tip_curl: 0.25,
                asymmetry: -0.2,
            },
            // Variant 7: very pronounced bend and curl
            GrassBladeVariant {
                bend_forward: 0.5,
                tip_curl: 0.4,
                asymmetry: 0.15,
            },
        ]
    }
}

/// Generate a grass blade mesh with the given variant parameters.
///
/// Returns a Bevy Mesh suitable for rendering via instancing.
pub fn generate_grass_blade(config: &GrassBladeMeshConfig, variant: GrassBladeVariant) -> Mesh {
    let segments = config.segments as usize;
    let height = config.blade_height;
    let width = config.blade_width;

    // Vertices: 2 columns (left and right side of blade) × (segments + 1) rows
    // + 2 bottom vertices = 2 * (segments + 1) + 2 = 2*segments + 4 total
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();

    // Bottom vertices (not used yet, but kept for reference)
    let _bottom_center = [0.0, 0.0, 0.0];
    let _bottom_left = [-width * 0.5, 0.0, 0.0];
    let _bottom_right = [width * 0.5, 0.0, 0.0];

    // Generate vertices along the blade height
    for seg in 0..=segments {
        let t = seg as f32 / segments as f32; // 0..1 along blade height

        // Forward bend: quadratic ease-out curve
        let bend = variant.bend_forward * (1.0 - (1.0 - t) * (1.0 - t)) * height * 0.5;

        // Tip curl: increases toward the tip
        let curl = variant.tip_curl * t * std::f32::consts::PI * 0.25; // Up to 45° curl

        // Asymmetry: slight offset to one side
        let asymmetry_offset = variant.asymmetry * width * t;

        // Taper: blade gets narrower toward the tip
        let taper = 1.0 - t * 0.6; // 100% at base, 40% at tip

        // Left vertex
        let left_x = -width * 0.5 * taper + asymmetry_offset * 0.5;
        let left_z = curl * width * 0.25; // Curl creates Z movement
        let left_y = t * height + bend;
        positions.push([left_x, left_y, left_z]);

        // Right vertex
        let right_x = width * 0.5 * taper + asymmetry_offset * 0.5;
        let right_z = curl * width * 0.25;
        let right_y = t * height + bend;
        positions.push([right_x, right_y, right_z]);

        // Normals: approximate cross-section normals (facing outward per side)
        // This is simplified; real implementation might compute per-face normals
        let _normal_angle = curl;
        normals.push([-0.707, 0.0, 0.0]); // Left face
        normals.push([0.707, 0.0, 0.0]); // Right face

        // UVs: v = position along blade, u = left/right side
        uvs.push([0.0, t]);
        uvs.push([1.0, t]);
    }

    // Indices: two-sided quad mesh
    // Front-facing quads (CCW from front): left top, left bottom, right bottom, right top
    let mut indices: Vec<u32> = Vec::new();

    for seg in 0..segments {
        let base = (seg * 2) as u32;
        let next_base = ((seg + 1) * 2) as u32;

        // Front face (CCW): (left, left_next, right_next), (left, right_next, right)
        indices.push(base); // left_current
        indices.push(next_base); // left_next
        indices.push(next_base + 1); // right_next

        indices.push(base); // left_current
        indices.push(next_base + 1); // right_next
        indices.push(base + 1); // right_current

        // Back face (CW = reversed for back): same triangle, opposite winding
        indices.push(base); // left_current
        indices.push(next_base + 1); // right_next
        indices.push(next_base); // left_next

        indices.push(base); // left_current
        indices.push(base + 1); // right_current
        indices.push(next_base + 1); // right_next
    }

    // Build mesh
    let mut mesh = Mesh::new(
        bevy::mesh::PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(bevy::mesh::Indices::U32(indices));

    mesh
}

/// Generate all 8 grass blade variants.
pub fn generate_grass_blade_variants(config: &GrassBladeMeshConfig) -> Vec<Mesh> {
    let variants = GrassBladeVariant::create_variants();
    variants
        .iter()
        .map(|v| generate_grass_blade(config, *v))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grass_blade_variants() {
        let variants = GrassBladeVariant::create_variants();
        assert_eq!(variants.len(), 8);

        // Check that variants have reasonable ranges
        for v in variants {
            assert!(v.bend_forward >= 0.0 && v.bend_forward <= 1.0);
            assert!(v.tip_curl >= 0.0 && v.tip_curl <= 1.0);
            assert!(v.asymmetry >= -1.0 && v.asymmetry <= 1.0);
        }
    }

    #[test]
    fn test_generate_grass_blade() {
        let config = GrassBladeMeshConfig::default();
        let variant = GrassBladeVariant::create_variants()[0];
        let mesh = generate_grass_blade(&config, variant);

        // Check that mesh has valid data
        assert!(mesh.attribute(Mesh::ATTRIBUTE_POSITION).is_some());
        assert!(mesh.attribute(Mesh::ATTRIBUTE_NORMAL).is_some());
        assert!(mesh.attribute(Mesh::ATTRIBUTE_UV_0).is_some());

        // With segments=4, we expect (4+1)*2 = 10 vertices
        if let Some(positions) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            assert_eq!(positions.len(), 10); // 5 segments * 2 sides
        }
    }

    #[test]
    fn test_generate_all_variants() {
        let config = GrassBladeMeshConfig::default();
        let meshes = generate_grass_blade_variants(&config);
        assert_eq!(meshes.len(), 8);

        for mesh in meshes {
            assert!(mesh.attribute(Mesh::ATTRIBUTE_POSITION).is_some());
        }
    }

    #[test]
    fn test_grass_blade_config_default() {
        let config = GrassBladeMeshConfig::default();
        assert!(config.blade_height > 0.0);
        assert!(config.blade_width > 0.0);
        assert!(config.segments > 0);
    }
}
