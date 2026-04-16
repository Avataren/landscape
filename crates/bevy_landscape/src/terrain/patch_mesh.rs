use bevy::prelude::*;

/// Builds a reusable square grid patch mesh with `resolution` quads per edge.
///
/// Vertices are at local positions [0, 1] in XZ (Y = 0).
/// UVs match local XZ directly.
/// The same mesh is instanced for every patch in the clipmap.
pub fn build_patch_mesh(resolution: u32) -> Mesh {
    let verts_per_edge = resolution + 1;
    let total_verts = (verts_per_edge * verts_per_edge) as usize;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(total_verts);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(total_verts);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(total_verts);

    for z in 0..verts_per_edge {
        for x in 0..verts_per_edge {
            let u = x as f32 / resolution as f32;
            let v = z as f32 / resolution as f32;
            positions.push([u, 0.0, v]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([u, v]);
        }
    }

    let quad_count = (resolution * resolution) as usize;
    let mut indices: Vec<u32> = Vec::with_capacity(quad_count * 6);

    for z in 0..resolution {
        for x in 0..resolution {
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
    fn patch_vertex_count() {
        let m = build_patch_mesh(4);
        let pos = m.attribute(Mesh::ATTRIBUTE_POSITION).unwrap();
        // (4+1)^2 = 25
        assert_eq!(pos.len(), 25);
    }

    #[test]
    fn patch_index_count() {
        let m = build_patch_mesh(4);
        // 4*4 quads * 6 indices = 96
        if let Some(bevy::mesh::Indices::U32(idx)) = m.indices() {
            assert_eq!(idx.len(), 96);
        } else {
            panic!("expected U32 indices");
        }
    }
}
