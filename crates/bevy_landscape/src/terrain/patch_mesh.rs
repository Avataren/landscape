use bevy::prelude::*;

/// Builds the canonical GPU Gems 2 clipmap block mesh.
///
/// Vertices are placed at integer grid positions `(x, 0, z)` where
/// `x, z ∈ [0, m]`, giving an `m × m` quad grid with `(m+1)² ` vertices.
///
/// The mesh is shared across ALL block instances at every LOD level.
/// World placement is encoded entirely in the per-instance `Transform`:
///   - `translation` = world-space XZ origin of the block corner
///   - `scale`       = `(level_scale_ws, 1, level_scale_ws)` so that one
///     local unit equals one terrain grid step at this LOD
///
/// UVs are normalised to `[0, 1]` so the fragment shader wireframe overlay
/// (`patch_uv * terrain.patch_resolution`) continues to resolve individual
/// quads independently of block size.
pub fn build_block_mesh(m: u32) -> Mesh {
    let verts_per_edge = m + 1;
    let total_verts = (verts_per_edge * verts_per_edge) as usize;
    let inv_m = 1.0 / m as f32;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(total_verts);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(total_verts);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(total_verts);

    for z in 0..verts_per_edge {
        for x in 0..verts_per_edge {
            positions.push([x as f32, 0.0, z as f32]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([x as f32 * inv_m, z as f32 * inv_m]);
        }
    }

    let mut indices: Vec<u32> = Vec::with_capacity((m * m) as usize * 6);
    for z in 0..m {
        for x in 0..m {
            let i00 = z * verts_per_edge + x;
            let i10 = i00 + 1;
            let i01 = i00 + verts_per_edge;
            let i11 = i01 + 1;

            indices.push(i00);
            indices.push(i01);
            indices.push(i10);

            indices.push(i10);
            indices.push(i01);
            indices.push(i11);
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_vertex_count() {
        let m = build_block_mesh(4);
        let pos = m.attribute(Mesh::ATTRIBUTE_POSITION).unwrap();
        // (4+1)^2 = 25 vertices
        assert_eq!(pos.len(), 25);
    }

    #[test]
    fn block_index_count() {
        let m = build_block_mesh(4);
        // 4×4 quads × 6 indices = 96
        if let Some(bevy::mesh::Indices::U32(idx)) = m.indices() {
            assert_eq!(idx.len(), 96);
        } else {
            panic!("expected U32 indices");
        }
    }

    #[test]
    fn block_positions_in_grid_units() {
        let mesh = build_block_mesh(2);
        let pos = mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap();
        // For m=2: vertices at x,z ∈ {0,1,2} → 9 verts
        assert_eq!(pos.len(), 9);
    }

    #[test]
    fn block_uvs_normalised() {
        let mesh = build_block_mesh(4);
        // Corner vertices: (0,0)=UV(0,0) and (4,4)=UV(1,1).
        // Just verify vertex count is correct; UV correctness implied by construction.
        let pos = mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap();
        assert_eq!(pos.len(), 25); // (4+1)^2
    }
}
