//! Painted foliage splatmap I/O.
//!
//! Stores and loads per-tile painted foliage density (0-255 per pixel).
//! Format: raw u8 values (256×256 = 65536 bytes per L0 tile).

use std::path::Path;

/// Read a painted splatmap tile (R8 density, 256×256 pixels).
///
/// Returns a Vec of 256×256 = 65,536 u8 values, row-major order.
/// Returns zeros if the file doesn't exist (treated as "no painting").
pub fn read_painted_splatmap(
    path: &Path,
    tile_size: u32,
) -> std::io::Result<Vec<u8>> {
    let expected_size = (tile_size * tile_size) as usize;

    // If file doesn't exist, return zeros (equivalent to "unpainted")
    if !path.exists() {
        return Ok(vec![0u8; expected_size]);
    }

    let data = std::fs::read(path)?;
    if data.len() != expected_size {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "painted splatmap has {} bytes, expected {} ({}×{})",
                data.len(),
                expected_size,
                tile_size,
                tile_size
            ),
        ));
    }
    Ok(data)
}

/// Write a painted splatmap tile (R8 density).
pub fn write_painted_splatmap(path: &Path, splatmap: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, splatmap)?;
    Ok(())
}

/// Helper for managing painted splatmap tiles.
pub struct PaintedSplatmapManager {
    pub foliage_root: std::path::PathBuf,
}

impl PaintedSplatmapManager {
    pub fn new(foliage_root: impl AsRef<Path>) -> Self {
        Self {
            foliage_root: foliage_root.as_ref().to_path_buf(),
        }
    }

    /// Read painted splatmap for a specific mip level and tile coordinates.
    ///
    /// Returns a Vec of tile_size × tile_size u8 values.
    /// Returns zeros if the file doesn't exist (unpainted).
    pub fn read_tile(
        &self,
        mip_level: u8,
        tx: i32,
        ty: i32,
        tile_size: u32,
    ) -> std::io::Result<Vec<u8>> {
        let path = crate::foliage::painted_splatmap_path(&self.foliage_root, mip_level, tx, ty);
        read_painted_splatmap(&path, tile_size)
    }

    /// Write painted splatmap for a specific mip level and tile coordinates.
    pub fn write_tile(
        &self,
        mip_level: u8,
        tx: i32,
        ty: i32,
        splatmap: &[u8],
    ) -> std::io::Result<()> {
        let path = crate::foliage::painted_splatmap_path(&self.foliage_root, mip_level, tx, ty);
        write_painted_splatmap(&path, splatmap)
    }

    /// Clear painted splatmap for a specific tile (delete the file).
    pub fn clear_tile(
        &self,
        mip_level: u8,
        tx: i32,
        ty: i32,
    ) -> std::io::Result<()> {
        let path = crate::foliage::painted_splatmap_path(&self.foliage_root, mip_level, tx, ty);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_read_nonexistent_returns_zeros() {
        let temp_dir = TempDir::new().unwrap();
        let nonexistent = temp_dir.path().join("nonexistent.bin");

        let data = read_painted_splatmap(&nonexistent, 256).unwrap();
        assert_eq!(data.len(), 256 * 256);
        assert!(data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_write_and_read() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test_splatmap.bin");

        // Create test data: 256×256 tile
        let tile_size = 256u32;
        let mut data = vec![0u8; (tile_size * tile_size) as usize];
        // Set some pixels to non-zero values
        data[0] = 255;
        data[100] = 128;
        data[(256 * 256 - 1) as usize] = 64;

        // Write
        write_painted_splatmap(&path, &data).unwrap();
        assert!(path.exists());

        // Read back
        let read_data = read_painted_splatmap(&path, tile_size).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_read_wrong_size_fails() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("wrong_size.bin");

        // Write wrong-sized data
        std::fs::write(&path, vec![0u8; 1000]).unwrap();

        // Read should fail
        let result = read_painted_splatmap(&path, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_manager_read_write() {
        let temp_dir = TempDir::new().unwrap();
        let manager = PaintedSplatmapManager::new(temp_dir.path());

        // Create test data
        let tile_size = 256u32;
        let mut data = vec![0u8; (tile_size * tile_size) as usize];
        data[0] = 200;
        data[1000] = 150;

        // Write via manager
        manager.write_tile(2, 5, -3, &data).unwrap();

        // Read back via manager
        let read_data = manager.read_tile(2, 5, -3, tile_size).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_manager_clear_tile() {
        let temp_dir = TempDir::new().unwrap();
        let manager = PaintedSplatmapManager::new(temp_dir.path());

        let tile_size = 256u32;
        let data = vec![100u8; (tile_size * tile_size) as usize];

        // Write
        manager.write_tile(2, 5, -3, &data).unwrap();
        let path = crate::foliage::painted_splatmap_path(temp_dir.path(), 2, 5, -3);
        assert!(path.exists());

        // Clear
        manager.clear_tile(2, 5, -3).unwrap();
        assert!(!path.exists());

        // Reading cleared tile should return zeros
        let zeros = manager.read_tile(2, 5, -3, tile_size).unwrap();
        assert!(zeros.iter().all(|&b| b == 0));
    }
}
