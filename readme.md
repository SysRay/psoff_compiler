# Structure
## IR
General IR struct (for frontend and SSA)

## frontend
* Parse binary and create frontend-ir (ir_types).
* Analysis: Calculate branches and create a CFG (with registers). 

If done, transform into a RSVDG, leaving frontend.

## Analysis
SCC
