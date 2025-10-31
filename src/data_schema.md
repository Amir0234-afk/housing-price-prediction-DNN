## Expected schema

Columns:
- Space(M^2)
- Inverse of space(1/m^2)
- Age(year)
- room(int)
- floor(int)
- elevator(bool)
- parking(bool)
- storage(bool)
- single unit in floor(bool)
- balkony(bool)
- lobby(bool)
- Price(Mil/M^2)  ← target required for training. Omit for pure prediction files.

Notes:
- Booleans may be 0/1, true/false, or Yes/No. They will be coerced to 0/1.
- File format: CSV or Excel (.xlsx). For Excel, the first sheet is read.
