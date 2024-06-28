!> Calculates dynamic models similar to Germano 1991 et. al,
!! following work of Perezhogin & Glazunov 2023
!! Implemented by Perezhogin P.A. Contact: pperezhogin@gmail.com
module MOM_dynamic_closures

! This file is part of MOM6. See LICENSE.md for the license.
use MOM_grid,          only : ocean_grid_type
use MOM_verticalGrid,  only : verticalGrid_type
use MOM_diag_mediator, only : diag_ctrl, time_type
use MOM_file_parser,   only : get_param, log_version, param_file_type
use MOM_unit_scaling,  only : unit_scale_type
use MOM_diag_mediator, only : post_data, register_diag_field
use MOM_domains,       only : create_group_pass, do_group_pass, group_pass_type, &
                              start_group_pass, complete_group_pass
use MOM_domains,       only : To_North, To_East
use MOM_domains,       only : pass_var, CORNER
use MOM_cpu_clock,     only : cpu_clock_id, cpu_clock_begin, cpu_clock_end
use MOM_cpu_clock,     only : CLOCK_MODULE, CLOCK_ROUTINE

implicit none ; private

#include <MOM_memory.h>

public PG23_germano_identity, PG23_init, PG23_end

!> Control structure for Perezhogin & Glazunov 2023
type, public :: PG23_CS ; private
  ! Parameters
  real :: test_width      !< Width of the test filter (hat) w.r.t. grid spacing
  real :: filters_ratio   !< The ratio of combined (hat(bar)) to base (bar) filters
  
  type(diag_ctrl), pointer :: diag => NULL() !< A type that regulates diagnostics output
  !>@{ Diagnostic handles
  !integer :: id_smt = -1
  !>@}

  !>@{ CPU time clock IDs
  integer :: id_clock_module
  !>@}
end type PG23_CS

contains

!> Read parameters, allocate and precompute arrays,
!! register diagnosicts
subroutine PG23_init(Time, G, GV, US, param_file, diag, CS, use_PG23)
  type(time_type),         intent(in)    :: Time       !< The current model time.
  type(ocean_grid_type),   intent(in)    :: G          !< The ocean's grid structure.
  type(verticalGrid_type), intent(in)    :: GV         !< The ocean's vertical grid structure
  type(unit_scale_type),   intent(in)    :: US         !< A dimensional unit scaling type
  type(param_file_type),   intent(in)    :: param_file !< Parameter file parser structure.
  type(diag_ctrl), target, intent(inout) :: diag       !< Diagnostics structure.
  type(PG23_CS),           intent(inout) :: CS         !< PG23 control structure.
  logical,                 intent(out)   :: use_PG23   !< If true, turns on ZB scheme.

  ! This include declares and sets the variable "version".
#include "version_variable.h"
  character(len=40)  :: mdl = "MOM_dynamic_closures" ! This module's name.

  call log_version(param_file, mdl, version, "")

  call get_param(param_file, mdl, "USE_PG23", use_PG23, &
                 "If true, turns on Perezhogin & Glazunov 2023 dynamic closure of " //&
                 "subgrid momentum parameterization of mesoscale eddies.", default=.false.)
  if (.not. use_PG23) return

  call get_param(param_file, mdl, "PG_TEST_WIDTH", CS%test_width, &
                 "Width of the test filter (hat) w.r.t. grid spacing", units="nondim", default=SQRT(6.0))

  call get_param(param_file, mdl, "PG_FILTERS_RATIO", CS%filters_ratio, &
                 "The ratio of combined (hat(bar)) to base (bar) filters", units="nondim", default=SQRT(2.0))

  ! Register fields for output from this module.
  !CS%diag => diag

  ! Clock IDs
  CS%id_clock_module = cpu_clock_id('(Ocean Perezhogin-Glazunov-2023)', grain=CLOCK_MODULE)

end subroutine PG23_init

!> Deallocate any variables allocated in PG23_init
subroutine PG23_end(CS)
  type(PG23_CS), intent(inout) :: CS  !< PG23 control structure.

end subroutine PG23_end

!> This function estimates free parameters of PG23 closure 
!! using germano identity
subroutine PG23_germano_identity(u, v, h, G, GV, CS)
type(ocean_grid_type),         intent(in)    :: G  !< The ocean's grid structure.
type(verticalGrid_type),       intent(in)    :: GV !< The ocean's vertical grid structure.
type(PG23_CS),                 intent(inout) :: CS  !< PG23 control structure.

real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
      intent(in)    :: u  !< The zonal velocity [L T-1 ~> m s-1].
real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
      intent(in)    :: v  !< The meridional velocity [L T-1 ~> m s-1].
real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  &
      intent(in)    :: h  !< Layer thicknesses [H ~> m or kg m-2].

call cpu_clock_begin(CS%id_clock_module)

call cpu_clock_end(CS%id_clock_module)

end subroutine PG23_germano_identity

end module MOM_dynamic_closures