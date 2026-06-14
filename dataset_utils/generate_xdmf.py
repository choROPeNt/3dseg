#!/usr/bin/env python3
"""Generate XDMF visualization files from 3D segmentation predictions."""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Mapping, Optional
import argparse
import h5py as h5
import numpy as np


def write_xdmf_for_h5(
    h5_path: str | Path,
    keys: Iterable[str],
    *,
    grid_key: str = "volume",
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    center: str = "Node",
    make_vec_vds_if_needed: bool = True,
    vds_suffix: str = "_zyx3",
    attribute_type_override: Optional[Mapping[str, str]] = None,
) -> Path:
    """
    Create XDMF file referencing selected datasets.

    Parameters
    ----------
    h5_path : str | Path
        Path to HDF5 file
    keys : Iterable[str]
        Dataset keys to include in XDMF
    grid_key : str
        Key of the grid dataset defining dimensions
    spacing_xyz : (dx, dy, dz)
        Physical voxel spacing
    origin_xyz : (ox, oy, oz)
        Physical origin
    center : {'Node', 'Cell'}
        Whether data is at nodes or cells
    make_vec_vds_if_needed : bool
        Create virtual datasets for vector data
    vds_suffix : str
        Suffix for virtual dataset names
    attribute_type_override : dict, optional
        Override attribute types for specific keys
    """

    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    if center not in {"Node", "Cell"}:
        raise ValueError("center must be 'Node' or 'Cell'")

    keys = list(keys)
    attribute_type_override = dict(attribute_type_override or {})

    dx, dy, dz = map(float, spacing_xyz)
    ox, oy, oz = map(float, origin_xyz)

    with h5.File(h5_path, "a" if make_vec_vds_if_needed else "r") as f:
        if grid_key not in f:
            raise KeyError(f"Missing grid_key '{grid_key}'")

        grid = f[grid_key]

        if not isinstance(grid, h5.Dataset):
            raise TypeError(f"'{grid_key}' is not an h5.Dataset")

        if grid.ndim < 3:
            raise ValueError(f"'{grid_key}' must have at least 3 dims, got {grid.shape}")

        Z, Y, X = grid.shape

        if center == "Cell":
            topo_dims = f"{Z + 1} {Y + 1} {X + 1}"
        else:
            topo_dims = f"{Z} {Y} {X}"

        attr_blocks = []

        for key in keys:
            if key not in f:
                raise KeyError(f"Missing dataset '{key}'")

            ds = f[key]
            target_key = key
            atype = attribute_type_override.get(key)

            if not isinstance(ds, h5.Dataset):
                raise TypeError(f"'{key}' is not an h5.Dataset")

            if ds.ndim == 3 and ds.shape == (Z, Y, X):
                atype = atype or "Scalar"
                dims = f"{Z} {Y} {X}"

            elif ds.ndim == 4 and ds.shape == (Z, Y, X, 3):
                atype = atype or "Vector"
                dims = f"{Z} {Y} {X} 3"

            elif ds.ndim == 4 and ds.shape[0] == 3 and ds.shape[1:] == (Z, Y, X):
                atype = atype or "Vector"
                dims = f"{Z} {Y} {X} 3"

                if not make_vec_vds_if_needed:
                    raise ValueError(f"{key} is (3,Z,Y,X) but VDS disabled.")

                vds_key = f"{key}{vds_suffix}"
                if vds_key in f:
                    del f[vds_key]

                layout = h5.VirtualLayout(shape=(Z, Y, X, 3), dtype=ds.dtype)
                vsrc = h5.VirtualSource(ds)
                for c in range(3):
                    layout[:, :, :, c] = vsrc[c, :, :, :]
                f.create_virtual_dataset(vds_key, layout)

                target_key = vds_key

            else:
                raise ValueError(f"Unsupported shape for '{key}': {ds.shape}")

            attr_blocks.append(
                f"""
      <Attribute Name="{key}" AttributeType="{atype}" Center="{center}">
        <DataItem Dimensions="{dims}" Format="HDF">
          {h5_path.name}:/{target_key}
        </DataItem>
      </Attribute>""".rstrip()
            )

    xmf_path = h5_path.with_suffix(".xdmf")

    xml = f"""<?xml version="1.0" ?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="ImageData" GridType="Uniform">
      <Topology TopologyType="3DCoRectMesh" Dimensions="{topo_dims}"/>
      <Geometry GeometryType="ORIGIN_DXDYDZ">
        <DataItem Dimensions="3" Format="XML">{oz} {oy} {ox}</DataItem>
        <DataItem Dimensions="3" Format="XML">{dz} {dy} {dx}</DataItem>
      </Geometry>
{chr(10).join(attr_blocks)}
    </Grid>
  </Domain>
</Xdmf>
"""

    xmf_path.write_text(xml, encoding="utf-8")
    return xmf_path


def process_predictions(pred_file: Path, crop_size: int = 512) -> None:
    """
    Process predictions file and generate XDMF visualizations.

    Parameters
    ----------
    pred_file : Path
        Path to .predictions.h5 file
    crop_size : int
        Size of center crop for quick preview (default 512)
    """
    pred_file = Path(pred_file)
    if not pred_file.exists():
        raise FileNotFoundError(f"File not found: {pred_file}")

    output_dir = pred_file.parent
    base_name = pred_file.stem.replace(".vol.pred", "")

    # Output files
    labels_h5 = output_dir / f"{base_name}.labels.h5"
    labels_crop_h5 = output_dir / f"{base_name}.labels.crop.h5"

    print(f"Processing: {pred_file}")
    print(f"Output directory: {output_dir}")

    # === Generate full labels dataset ===
    print("\n1. Generating full labels dataset...")
    with h5.File(pred_file, "r") as fin, h5.File(labels_h5, "w") as fout:
        pred = fin["predictions"]

        labels = fout.create_dataset(
            "labels",
            shape=(600, 2200, 2200),  # (Z, Y, X)
            dtype=np.uint8,
            chunks=(19, 69, 69),
            compression="gzip"
        )

        _, nz, ny, nx = pred.shape
        _, cz, cy, cx = pred.chunks

        for z0 in range(0, nz, cz):
            z1 = min(z0 + cz, nz)
            for y0 in range(0, ny, cy):
                y1 = min(y0 + cy, ny)
                for x0 in range(0, nx, cx):
                    x1 = min(x0 + cx, nx)

                    block = pred[:, z0:z1, y0:y1, x0:x1]
                    labels[z0:z1, y0:y1, x0:x1] = np.argmax(block, axis=0)

                    print(f"  z={z0}:{z1}, y={y0}:{y1}, x={x0}:{x1}", end="\r")

    print(f"✓ Written: {labels_h5}")

    # === Generate XDMF for full dataset ===
    print("\n2. Generating XDMF for full dataset...")
    xdmf_path = write_xdmf_for_h5(
        labels_h5,
        keys=["labels"],
        grid_key="labels",
        center="Cell",
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_xyz=(0.0, 0.0, 0.0),
    )
    print(f"✓ Written: {xdmf_path}")

    # === Generate cropped dataset ===
    print(f"\n3. Generating center-cropped dataset ({crop_size}×{crop_size}×600)...")
    full_size = 2200
    start_idx = (full_size - crop_size) // 2
    end_idx = start_idx + crop_size

    with h5.File(labels_h5, "r") as fin, h5.File(labels_crop_h5, "w") as fout:
        labels_full = fin["labels"]
        labels_crop = fout.create_dataset(
            "labels",
            shape=(600, crop_size, crop_size),  # (Z, Y, X)
            dtype=np.uint8,
            chunks=(19, 69, 69),
            compression="gzip",
        )
        labels_crop[:, :, :] = labels_full[:, start_idx:end_idx, start_idx:end_idx]

    print(f"✓ Written: {labels_crop_h5}")

    # === Generate XDMF for cropped dataset ===
    print("4. Generating XDMF for cropped dataset...")
    xdmf_crop_path = write_xdmf_for_h5(
        labels_crop_h5,
        keys=["labels"],
        grid_key="labels",
        center="Cell",
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_xyz=(0.0, 0.0, 0.0),
    )
    print(f"✓ Written: {xdmf_crop_path}")

    # === Summary ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Full dataset:   {labels_h5.name}")
    print(f"               {xdmf_path.name}")
    print(f"               Shape: (Z, Y, X) = (600, 2200, 2200)")
    print()
    print(f"Cropped preview: {labels_crop_h5.name}")
    print(f"                {xdmf_crop_path.name}")
    print(f"                Shape: (Z, Y, X) = (600, {crop_size}, {crop_size})")
    print()
    print("Open .xdmf files in ParaView for visualization.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate XDMF visualization files from 3D segmentation predictions."
    )
    parser.add_argument(
        "predictions_h5",
        type=str,
        help="Path to predictions.h5 file",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=512,
        help="Size of center crop for preview (default: 512)",
    )

    args = parser.parse_args()
    process_predictions(args.predictions_h5, crop_size=args.crop_size)
