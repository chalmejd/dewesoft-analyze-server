# dewesoft-analyze-server
Server for Python Scripts used by Dewesoft Analyze web app: https://github.com/chalmejd-TEAM/dewesoft-analyze

This package is based on the [DEWESoft Data Reader Library](https://dewesoft.com/download/developer-downloads).

## Testing dashboard backend

The Allen-Bradley and M700 testing dashboard backends are now hosted inside this Flask app.

Endpoints:

- `GET /api/testing-dashboard/ab/devices`
- `GET /api/testing-dashboard/ab/data?device=<name>`
- `GET /api/testing-dashboard/m700/devices`
- `GET /api/testing-dashboard/m700/registers?device=<name>&register=<index>`

Notes:

- The backend requires `pymodbus` in addition to the existing Flask dependencies.
- AB configuration is stored in `testing_dashboard_config/ab_devices.csv` and `testing_dashboard_config/ab_register_map.csv`.
- These routes preserve the JSON payloads expected by the React testing dashboard page.
