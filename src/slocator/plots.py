"""Interactive plotting for territory optimization data.

Creates Plotly choropleth and scatter maps from GeoJSON files (local or HTTP).
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import aiohttp
import pandas as pd
import plotly.express as px


class InteractivePlotter:
    """Handle interactive plotting for territory optimization data."""

    def __init__(self):
        self.grid_df = pd.DataFrame()
        self.supermarkets_df = pd.DataFrame()
        self.grid_geojson = None
        self.places_geojson = None
        self.data_loaded = False

    def _load_geojson(self, file_path: str) -> Optional[dict]:
        try:
            if file_path.startswith(("http://", "https://")):
                print(f"[DEBUG] Fetching GeoJSON from URL: {file_path}")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                geojson_data = loop.run_until_complete(self._fetch_url(file_path))
                loop.close()
                return geojson_data
            if Path(file_path).exists():
                print(f"[DEBUG] Reading GeoJSON from local file: {file_path}")
                with open(file_path) as f:
                    return json.load(f)
            print(f"❌ Local file not found: {file_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading GeoJSON from {file_path}: {str(e)}")
            return None

    async def _fetch_url(self, url: str) -> Optional[dict]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    print(f"❌ HTTP error {response.status} fetching {url}")
                    return None
        except Exception as e:
            print(f"❌ Error fetching URL {url}: {str(e)}")
            return None

    def load_data_files(self, data_files: Dict[str, str]) -> bool:
        try:
            print(f"[DEBUG] Loading data files for interactive plotting: {data_files}")

            grid_path = data_files.get("grid_data")
            if not grid_path:
                print("❌ No grid_data path provided")
                return False
            self.grid_geojson = self._load_geojson(grid_path)
            if not self.grid_geojson:
                print(f"❌ Failed to load grid data from: {grid_path}")
                return False
            self.grid_df = pd.DataFrame([f["properties"] for f in self.grid_geojson["features"]])
            self.grid_df["id"] = [f["id"] for f in self.grid_geojson["features"]]
            print(f"✅ Grid data loaded: {len(self.grid_df)} features")

            places_path = data_files.get("places_data")
            if not places_path:
                print("❌ No places_data path provided")
                return False
            self.places_geojson = self._load_geojson(places_path)
            if not self.places_geojson:
                print(f"❌ Failed to load places data from: {places_path}")
                return False
            places_df = pd.DataFrame([f["properties"] for f in self.places_geojson["features"]])
            places_df["lon"] = [f["geometry"]["coordinates"][0] for f in self.places_geojson["features"]]
            places_df["lat"] = [f["geometry"]["coordinates"][1] for f in self.places_geojson["features"]]
            self.supermarkets_df = places_df[places_df["primaryType"] == "supermarket"].reset_index(drop=True)
            print(f"✅ Places data loaded: {len(self.supermarkets_df)} supermarkets")

            self.data_loaded = True
            return True

        except Exception as e:
            print(f"❌ Error loading data files: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def get_available_variables(self) -> Dict[str, str]:
        if not self.data_loaded or self.grid_df.empty:
            return {}
        column_mappings = {
            "number_of_persons": "Number of Persons",
            "population_purchasing_power": "Population Purchasing Power",
            "number_of_supermarkets": "Number of Supermarkets",
            "population_purchasing_potential": "Population Purchasing Potential",
        }
        return {k: v for k, v in column_mappings.items() if k in self.grid_df.columns}

    def create_choropleth_map(self, selected_variable: str) -> Optional[dict]:
        if not self.data_loaded or self.grid_df.empty or self.grid_geojson is None:
            print("❌ Cannot create choropleth: data not loaded")
            return None
        if selected_variable not in self.grid_df.columns:
            print(f"❌ Variable '{selected_variable}' not found in data")
            return None
        try:
            fig = px.choropleth_mapbox(
                self.grid_df,
                geojson=self.grid_geojson,
                locations="id",
                featureidkey="id",
                color=selected_variable,
                color_continuous_scale="Plasma",
                mapbox_style="carto-positron",
                zoom=9,
                center={"lat": 24.7, "lon": 46.7},
                opacity=0.6,
                labels={selected_variable: selected_variable.replace("_", " ").title()},
                hover_data={
                    "id": False,
                    **{
                        col: ":.2s" if "power" in col or "potential" in col else True
                        for col in self.grid_df.columns
                        if col != "id"
                    },
                },
            )
            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=500)
            return fig
        except Exception as e:
            print(f"❌ Error creating choropleth map: {str(e)}")
            return None

    def create_supermarket_scatter_map(self) -> Optional[dict]:
        if not self.data_loaded or self.supermarkets_df.empty:
            print("❌ Cannot create scatter map: supermarket data not loaded")
            return None
        try:
            fig = px.scatter_mapbox(
                self.supermarkets_df,
                lat="lat",
                lon="lon",
                hover_name="name",
                hover_data={"name": True, "phone": True, "lon": False, "lat": False},
                color_discrete_sequence=["#1f77b4"],
                mapbox_style="carto-positron",
                zoom=9,
                center={"lat": 24.7, "lon": 46.7},
                title="Supermarket Locations",
            )
            fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=500)
            return fig
        except Exception as e:
            print(f"❌ Error creating scatter map: {str(e)}")
            return None

    def get_data_summary(self) -> Dict[str, object]:
        if not self.data_loaded:
            return {"status": "No data loaded"}
        return {
            "status": "Data loaded successfully",
            "grid_features": len(self.grid_df),
            "supermarkets": len(self.supermarkets_df),
            "available_variables": list(self.get_available_variables().keys()),
            "data_columns": list(self.grid_df.columns),
        }


plotter = InteractivePlotter()


def load_and_create_plots(data_files: Dict[str, str]) -> Tuple[bool, Dict[str, object]]:
    success = plotter.load_data_files(data_files)
    if not success:
        return False, {"error": "Failed to load data files"}

    available = plotter.get_available_variables()
    return True, {
        "variables": available,
        "summary": plotter.get_data_summary(),
        "default_variable": next(iter(available), None),
    }