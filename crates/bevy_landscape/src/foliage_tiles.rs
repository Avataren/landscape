//! Binary tile I/O for foliage instances.
//!
//! Stores and loads foliage instances per tile and LOD tier.
//! Format: u32 instance count, followed by `FoliageInstance` binary data (48 bytes each).

use crate::foliage::{FoliageInstance, FoliageLodTier};
use std::path::Path;

/// Header for a foliage tile: instance count (u32, little-endian).
const FOLIAGE_TILE_HEADER_SIZE: usize = 4;
/// Size of one serialized instance.
const FOLIAGE_INSTANCE_SIZE: usize = 48;

/// Serialize a list of foliage instances to binary format.
///
/// Format:
/// - Bytes 0-3: u32 instance count (little-endian)
/// - Bytes 4+: instance data (48 bytes per instance)
pub fn serialize_foliage_tile(instances: &[FoliageInstance]) -> Vec<u8> {
    let mut data = Vec::with_capacity(FOLIAGE_TILE_HEADER_SIZE + instances.len() * FOLIAGE_INSTANCE_SIZE);

    // Write header
    let count = instances.len() as u32;
    data.extend_from_slice(&count.to_le_bytes());

    // Write instances
    for inst in instances {
        data.extend_from_slice(&inst.to_bytes());
    }

    data
}

/// Deserialize foliage instances from binary format.
///
/// Expects format:
/// - Bytes 0-3: u32 instance count
/// - Bytes 4+: instance data (48 bytes per instance)
pub fn deserialize_foliage_tile(data: &[u8]) -> std::io::Result<Vec<FoliageInstance>> {
    if data.len() < FOLIAGE_TILE_HEADER_SIZE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "foliage tile too small: {} bytes, need at least {}",
                data.len(),
                FOLIAGE_TILE_HEADER_SIZE
            ),
        ));
    }

    let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let expected_size = FOLIAGE_TILE_HEADER_SIZE + count * FOLIAGE_INSTANCE_SIZE;

    if data.len() < expected_size {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "foliage tile has {} bytes, expected {} for {} instances",
                data.len(),
                expected_size,
                count
            ),
        ));
    }

    let mut instances = Vec::with_capacity(count);
    for i in 0..count {
        let offset = FOLIAGE_TILE_HEADER_SIZE + i * FOLIAGE_INSTANCE_SIZE;
        let bytes = <&[u8; FOLIAGE_INSTANCE_SIZE]>::try_from(&data[offset..offset + FOLIAGE_INSTANCE_SIZE])
            .map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "instance data too short")
            })?;
        instances.push(FoliageInstance::from_bytes(bytes));
    }

    Ok(instances)
}

/// Write foliage instances to a binary file.
pub fn write_foliage_tile(path: &Path, instances: &[FoliageInstance]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let data = serialize_foliage_tile(instances);
    std::fs::write(path, data)?;
    Ok(())
}

/// Read foliage instances from a binary file.
pub fn read_foliage_tile(path: &Path) -> std::io::Result<Vec<FoliageInstance>> {
    let data = std::fs::read(path)?;
    deserialize_foliage_tile(&data)
}

/// Helper to construct and write instances for a specific LOD tier.
///
/// Includes helper path construction for the expected directory layout.
pub struct FoliageTileWriter {
    pub foliage_root: std::path::PathBuf,
}

impl FoliageTileWriter {
    pub fn new(foliage_root: impl AsRef<Path>) -> Self {
        Self {
            foliage_root: foliage_root.as_ref().to_path_buf(),
        }
    }

    /// Write instances for a specific LOD tier, mip level, and tile coordinates.
    pub fn write_lod_tile(
        &self,
        lod: FoliageLodTier,
        mip_level: u8,
        tx: i32,
        ty: i32,
        instances: &[FoliageInstance],
    ) -> std::io::Result<()> {
        let path = crate::foliage::foliage_tile_path(&self.foliage_root, lod, mip_level, tx, ty);
        write_foliage_tile(&path, instances)
    }

    /// Read instances for a specific LOD tier, mip level, and tile coordinates.
    pub fn read_lod_tile(
        &self,
        lod: FoliageLodTier,
        mip_level: u8,
        tx: i32,
        ty: i32,
    ) -> std::io::Result<Vec<FoliageInstance>> {
        let path = crate::foliage::foliage_tile_path(&self.foliage_root, lod, mip_level, tx, ty);
        read_foliage_tile(&path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::prelude::*;

    #[test]
    fn test_serialize_deserialize_empty() {
        let instances: Vec<FoliageInstance> = vec![];
        let data = serialize_foliage_tile(&instances);
        let restored = deserialize_foliage_tile(&data).unwrap();
        assert_eq!(restored.len(), 0);
    }

    #[test]
    fn test_serialize_deserialize_single() {
        let inst = FoliageInstance::new(
            Vec3::new(10.0, 20.0, 30.0),
            Quat::IDENTITY,
            Vec3::new(1.5, 2.0, 1.5),
            3,
        );
        let instances = vec![inst];
        let data = serialize_foliage_tile(&instances);
        let restored = deserialize_foliage_tile(&data).unwrap();

        assert_eq!(restored.len(), 1);
        assert!(restored[0].position.abs_diff_eq(inst.position, 1e-6));
        assert!(restored[0].scale.abs_diff_eq(inst.scale, 1e-6));
        assert_eq!(restored[0].variant_id, 3);
    }

    #[test]
    fn test_serialize_deserialize_many() {
        let mut instances = Vec::new();
        for i in 0..100 {
            instances.push(FoliageInstance::new(
                Vec3::new(i as f32, i as f32 + 10.0, i as f32 + 20.0),
                Quat::from_rotation_z(i as f32 * 0.1),
                Vec3::splat(1.0 + (i as f32 * 0.01)),
                (i % 8) as u32,
            ));
        }

        let data = serialize_foliage_tile(&instances);
        let restored = deserialize_foliage_tile(&data).unwrap();

        assert_eq!(restored.len(), instances.len());
        for (orig, rest) in instances.iter().zip(restored.iter()) {
            assert!(rest.position.abs_diff_eq(orig.position, 1e-6));
            assert!(rest.rotation.abs_diff_eq(orig.rotation, 1e-6));
            assert!(rest.scale.abs_diff_eq(orig.scale, 1e-6));
            assert_eq!(rest.variant_id, orig.variant_id);
        }
    }

    #[test]
    fn test_deserialize_invalid_header() {
        let data = vec![1, 2]; // Too short for header
        let result = deserialize_foliage_tile(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_truncated_instances() {
        let mut data = vec![0, 0, 0, 1]; // count = 1 instance
        data.extend_from_slice(&[0; 32]); // Only 32 bytes, need 48
        let result = deserialize_foliage_tile(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_tile_writer_reader() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let writer = FoliageTileWriter::new(temp_dir.path());

        let inst1 = FoliageInstance::new(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY, Vec3::ONE, 1);
        let inst2 = FoliageInstance::new(Vec3::new(4.0, 5.0, 6.0), Quat::IDENTITY, Vec3::ONE, 2);
        let instances = vec![inst1, inst2];

        // Write to LOD0
        writer
            .write_lod_tile(FoliageLodTier::Lod0, 2, 5, -3, &instances)
            .unwrap();

        // Read back
        let read_instances = writer
            .read_lod_tile(FoliageLodTier::Lod0, 2, 5, -3)
            .unwrap();

        assert_eq!(read_instances.len(), 2);
        assert_eq!(read_instances[0].variant_id, 1);
        assert_eq!(read_instances[1].variant_id, 2);
    }
}
