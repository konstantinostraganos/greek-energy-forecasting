"""ENTSO-E Transparency Platform API client.

Fetches actual electricity load and day-ahead forecasts
for the Greek bidding zone via the ENTSO-E Restful API.
"""

from datetime import datetime, timedelta
from xml.etree import ElementTree as ET

import pandas as pd
import requests
from loguru import logger

from energy_forecasting.config import settings


class EntsoEClient:
    """Client for the ENTSO-E Transparency Platform API.

    The API returns XML data which we parse into DataFrames.
    Two main data types:
    - Actual load (what really happened)
    - Day-ahead forecast (what ENTSO-E predicted)
    """

    # ENTSO-E document type codes
    ACTUAL_LOAD = "A65"
    FORECAST_LOAD = "A65"

    # Process type codes
    REALISED = "A16"
    DAY_AHEAD = "A01"

    def __init__(self):
        """Initialize client with settings from config."""

        self.base_url = settings.entso_e.base_url
        self.area_code = settings.entso_e.area_code
        self.api_key = settings.entsoe_api_key

    def _fetch_data(
        self,
        document_type: str,
        process_type: str,
        start: datetime,
        end: datetime,
    ) -> str:
        """Send request to ENTSO-E API and return raw XML.

        This is the low-level method — other methods call this
        with the right document/process type codes.
        """

        if not self.api_key:
            raise ValueError("ENTSO-E API key not found. Add ENTSOE_API_KEY to your .env file.")

        # ENTSO-E expects dates as: 202001010000
        period_start = start.strftime("%Y%m%d%H%M")
        period_end = end.strftime("%Y%m%d%H%M")

        params = {
            "securityToken": self.api_key,
            "documentType": document_type,
            "processType": process_type,
            "outBiddingZone_Domain": self.area_code,
            "periodStart": period_start,
            "periodEnd": period_end,
        }

        logger.info(f"Fetching {process_type} data: {start.date()} to {end.date()}")

        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()

        return response.text

    def _parse_xml(self, xml_text: str) -> pd.DataFrame:
        """Parse ENTSO-E XML response into a DataFrame.

        Extracts timestamps and load values from the nested
        XML structure: TimeSeries → Period → Point
        """

        root = ET.fromstring(xml_text)

        # ENTSO-E XML uses namespaces — we need to handle this
        namespace = {"ns": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"}

        records = []

        for period in root.findall(".//ns:Period", namespace):
            # Get the start time of this period
            start_text = period.find("ns:timeInterval/ns:start", namespace).text
            period_start = datetime.fromisoformat(start_text.replace("Z", "+00:00"))

            # Get resolution (usually PT60M = 1 hour)
            resolution = period.find("ns:resolution", namespace).text
            if resolution == "PT60M":
                step = timedelta(hours=1)
            elif resolution == "PT15M":
                step = timedelta(minutes=15)
            else:
                step = timedelta(hours=1)

            # Extract each data point
            for point in period.findall("ns:Point", namespace):
                position = int(point.find("ns:position", namespace).text)
                quantity = float(point.find("ns:quantity", namespace).text)

                timestamp = period_start + (position - 1) * step

                records.append(
                    {
                        "timestamp": timestamp,
                        "load_mw": quantity,
                    }
                )

        df = pd.DataFrame(records)

        if df.empty:
            logger.warning("No data points found in XML response")
            return df

        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info(f"Parsed {len(df)} data points")

        return df

    def get_actual_load(
        self,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch actual electricity load for Greece.

        Args:
            start: Start datetime (UTC)
            end: End datetime (UTC)

        Returns:
            DataFrame with columns: timestamp, load_mw
        """

        xml = self._fetch_data(
            document_type=self.ACTUAL_LOAD,
            process_type=self.REALISED,
            start=start,
            end=end,
        )
        df = self._parse_xml(xml)
        df = df.rename(columns={"load_mw": "actual_load_mw"})
        return df

    def get_day_ahead_forecast(
        self,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch ENTSO-E day-ahead load forecast for Greece.

        This is the benchmark we compare our models against.

        Args:
            start: Start datetime (UTC)
            end: End datetime (UTC)

        Returns:
            DataFrame with columns: timestamp, forecast_load_mw
        """

        xml = self._fetch_data(
            document_type=self.FORECAST_LOAD,
            process_type=self.DAY_AHEAD,
            start=start,
            end=end,
        )
        df = self._parse_xml(xml)
        df = df.rename(columns={"load_mw": "forecast_load_mw"})
        return df

    def fetch_full_dataset(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch complete dataset: actual load + day-ahead forecast.

        The ENTSO-E API limits requests to ~1 year, so we split
        the date range into monthly chunks and merge everything.

        Args:
            start_date: "YYYY-MM-DD" or None (uses config)
            end_date: "YYYY-MM-DD" or None (uses config)

        Returns:
            DataFrame with columns: timestamp, actual_load_mw, forecast_load_mw
        """

        start = datetime.fromisoformat(start_date or settings.entso_e.start_date)
        end = datetime.fromisoformat(end_date or settings.entso_e.end_date)

        logger.info(f"Fetching full dataset: {start.date()} to {end.date()}")

        # Split into monthly chunks to avoid API limits
        chunks_actual = []
        chunks_forecast = []
        current = start

        while current < end:
            # Each chunk = 1 month
            chunk_end = min(
                current.replace(day=1) + timedelta(days=32),
                end,
            )
            # Ensure chunk_end starts at day 1 of next month
            if chunk_end != end:
                chunk_end = chunk_end.replace(day=1)

            try:
                actual = self.get_actual_load(current, chunk_end)
                chunks_actual.append(actual)
            except Exception as e:
                logger.error(
                    f"Failed to fetch actual load {current.date()} to {chunk_end.date()}: {e}"
                )

            try:
                forecast = self.get_day_ahead_forecast(current, chunk_end)
                chunks_forecast.append(forecast)
            except Exception as e:
                logger.error(
                    f"Failed to fetch forecast {current.date()} to {chunk_end.date()}: {e}"
                )

            current = chunk_end

        # Combine all chunks
        df_actual = (
            pd.concat(chunks_actual, ignore_index=True) if chunks_actual else pd.DataFrame()
        )
        df_forecast = (
            pd.concat(chunks_forecast, ignore_index=True) if chunks_forecast else pd.DataFrame()
        )

        # Merge actual and forecast on timestamp
        if not df_actual.empty and not df_forecast.empty:
            df = pd.merge(df_actual, df_forecast, on="timestamp", how="outer")
        elif not df_actual.empty:
            df = df_actual
        else:
            df = df_forecast

        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.drop_duplicates(subset="timestamp", keep="first")

        logger.info(
            f"Dataset complete: {len(df)} rows, {df['timestamp'].min()} to {df['timestamp'].max()}"
        )

        return df

    def save_raw_data(self, df: pd.DataFrame, filename: str = "entso_e_load.parquet") -> None:
        """Save raw data to Parquet format.

        Parquet is columnar, compressed, and preserves dtypes —
        much better than CSV for time series data.
        """

        save_dir = settings.data.raw_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        filepath = save_dir / filename
        df.to_parquet(filepath, index=False)

        size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {len(df)} rows to {filepath} ({size_mb:.1f} MB)")
