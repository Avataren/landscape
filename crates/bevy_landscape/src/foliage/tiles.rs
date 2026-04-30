//! Binary tile I/O for foliage instances.
//!
//! Format: u32 instance count, followed by `FoliageInstance` binary data (48 bytes each).

use super::{FoliageInstance, FoliageLodTier};
use std::path::Path;

const FOLIAGE_TILE_HEADER_SIZE: usize = 4;
const FOLIAGE_INSTANCE_SIZE: usize = 48;

/// Serialize a list of foliage instances to binary format.
pub fn serialize_foliage_tile(instances: &[FoliageInstance]) -> Vec<u8> {
    let mut data =
        Vec::with_capacity(FOLIAGE_TILE_HEADER_SIZE + instances.len() * FOLIAGE_INSTANCE_SIZE);
    let count = instances.len() as u32;
    data.extend_from_slice(&count.to_le_bytes());
    for inst in instances {
        data.extend_from_slice(&inst.to_bytes());
    }
    data
}

/// Deserialize foliage instances from binary format.
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
        let bytes =
            <&[u8; FOLIAGE_INSTANCE_SIZE]>::try_from(&data[offset..offset + FOLIAGE_INSTANCE_SIZE])
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
    std::fs::write(path, serialize_foliage_tile(instances))?;
    Ok(())
}

/// Read foliage instances from a binary file.
pub fn read_foliage_tile(path: &Path) -> std::io::Result<Vec<FoliageInstance>> {
    deserialize_foliage_tile(&std::fs::read(path)?)
}

/// Helper to construct and write instances for a specific LOD tier.
pub struct FoliageTileWriter {
    pub foliage_root: std::path::PathBuf,
}

impl FoliageTileWriter {
    pub fn new(foliage_root: impl AsRef<Path>) -> Self {
        Self { foliage_root: foliage_root.as_ref().to_path_buf() }
    }

    pub fn write_lod_tile(
        &self,
        lod: FoliageLodTier,
        mip_level: u8,
        tx: i32,
        ty: i32,
        instances: &[FoliageInstance],
    ) -> std::io::Result<()> {
        let path = super::foliage_tile_path(&self.foliage_root, lod, mip_level, tx, ty);
        write_foliage_tile(&path, instances)
    }

    pub fn read_lod_tile(
        &self,
        lod: FoliageLodTier,
        mip_level: u8,
        tx: i32,
        ty: i32,
    ) -> std::io::Result<Vec<FoliageInstance>> {
        let path = super::foliage_tile_path(&self.foliage_root, lod, mip_level, tx, ty);
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
        let data = serialize_foliage_tile(&[inst]);
        let restored = deserialize_foliage_tile(&data).unwrap();
        assert_eq!(restored.len(), 1);
        assert!(restored[0].position.abs_diff_eq(inst.position, 1e-6));
        assert_eq!(restored[0].variant_id, 3);
    }

    #[test]
    fn test_deserialize_invalid_header() {
        let result = deserialize_foliage_tile(&[1, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tile_writer_reader() {
        use tempfile::TempDir;
        let temp_dir = TempDir::new().unwrap();
        let writer = FoliageTileWriter::new(temp_dir.path());
        let instances = vec![
            FoliageInstance::new(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY, Vec3::ONE, 1),
            FoliageInstance::new(Vec3::new(4.0, 5.0, 6.0), Quat::IDENTITY, Vec3::ONE, 2),
        ];
        writer.write_lod_tile(FoliageLodTier::Lod0, 2, 5, -3, &instances).unwrap();
        let read_back = writer.read_lod_tile(FoliageLodTier::Lod0, 2, 5, -3).unwrap();
        assert_eq!(read_back.len(), 2);
        assert_eq!(read_back[0].variant_id, 1);
        assert_eq!(read_back[1].variant_id, 2);
    }
}
