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
use MOM_error_handler,         only : MOM_error, FATAL
use MOM_domains,       only : create_group_pass, do_group_pass, group_pass_type, &
                              start_group_pass, complete_group_pass
use MOM_coms,          only : reproducing_sum
use MOM_domains,       only : To_North, To_East
use MOM_domains,       only : pass_var, CORNER, pass_vector
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
  integer :: reduce       !< The reduction method in Germano identity
  logical :: offline      !< If offline, we only save fields but do not apply prediction
  logical, public :: ssm  !< Turns on the SSM model (i.e., Leonard Stress of Germano decomposition)
  logical, public :: reynolds  !< Turns on the Reynolds model (i.e., Reynolds Stress of Germano decomposition)
  logical :: zelong_dynamic !< Uses Zelong2022 dynamic procedure instead of Germano

  real, dimension(:,:), allocatable :: &
          dx_dyT,    & !< Pre-calculated dx/dy at h points [nondim]
          dy_dxT,    & !< Pre-calculated dy/dx at h points [nondim]
          dx_dyBu,   & !< Pre-calculated dx/dy at q points [nondim]
          dy_dxBu,   & !< Pre-calculated dy/dx at q points [nondim]
          grid_sp_q4   !< Pre-calculated dx^4  at q point  [L4 ~> m4]
  
  type(diag_ctrl), pointer :: diag => NULL() !< A type that regulates diagnostics output
  
  !>@{ Diagnostic handles
  integer :: id_u, id_v
  integer :: id_uf, id_vort_y, id_vort_yf, id_lap_vort_y, id_lap_vort_yf, id_smag_y, id_smag_yf, id_m_y, id_leo_y
  integer :: id_vf, id_vort_x, id_vort_xf, id_lap_vort_x, id_lap_vort_xf, id_smag_x, id_smag_xf, id_m_x, id_leo_x
  integer :: id_sh_xy, id_sh_xyf, id_vort_xy, id_vort_xyf, id_shear_mag, id_shear_magf, id_lap_vort, id_lap_vortf
  integer :: id_sh_xx, id_sh_xxf, id_lm, id_mm, id_mb, id_lb, id_bb
  integer :: id_smag, id_CR
  integer :: id_bx, id_by, id_bx_base, id_by_base
  integer :: id_h_x, id_h_y
  !>@}

  !>@{ CPU time clock IDs
  integer :: id_clock_module
  integer :: id_clock_mpi
  integer :: id_clock_filter
  integer :: id_clock_post
  integer :: id_clock_reduce
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

  integer :: i, j
  integer :: isd, ied, jsd, jed, IsdB, IedB, JsdB, JedB
  real :: dx2q, dy2q ! Squared grid spacings [L2 ~> m2]

  ! This include declares and sets the variable "version".
#include "version_variable.h"
  character(len=40)  :: mdl = "MOM_dynamic_closures" ! This module's name.

  isd  = G%isd  ; ied  = G%ied  ; jsd  = G%jsd  ; jed  = G%jed
  IsdB = G%IsdB ; IedB = G%IedB ; JsdB = G%JsdB ; JedB = G%JedB

  call log_version(param_file, mdl, version, "")

  call get_param(param_file, mdl, "USE_PG23", use_PG23, &
                 "If true, turns on Perezhogin & Glazunov 2023 dynamic closure of " //&
                 "subgrid momentum parameterization of mesoscale eddies.", default=.false.)
  if (.not. use_PG23) return

  call get_param(param_file, mdl, "PG23_TEST_WIDTH", CS%test_width, &
                 "Width of the test filter (hat) w.r.t. grid spacing", units="nondim", default=SQRT(6.0))

  call get_param(param_file, mdl, "PG23_FILTERS_RATIO", CS%filters_ratio, &
                 "The ratio of combined (hat(bar)) to base (bar) filters", units="nondim", default=SQRT(2.0))

  call get_param(param_file, mdl, "PG23_REDUCE", CS%reduce, &
                 "The reduction method in Germano identity. 0: sum, 1: sum of positive elements", default=0)

  call get_param(param_file, mdl, "PG23_OFFLINE", CS%offline, &
                 "If offline, we only save fields but do not apply prediction", default=.false.)

  call get_param(param_file, mdl, "PG23_SSM", CS%ssm, &
                 "Turns on the SSM model (i.e., Leonard Stress of Germano decomposition)", default=.false.)
  
  call get_param(param_file, mdl, "PG23_REYNOLDS", CS%reynolds, &
                 "Turns on the Reynolds model (i.e., Reynolds Stress of Germano decomposition)", default=.false.)

  call get_param(param_file, mdl, "PG23_ZELONG_DYNAMIC", CS%zelong_dynamic, &
                 "Uses Zelong2022 dynamic procedure instead of Germano", default=.false.)

  if ((CS%ssm .or. CS%zelong_dynamic) .and. ABS(CS%filters_ratio-SQRT(2.0))>1e-10) then
    call MOM_error(FATAL, &
      "MOM_dynamic_closures: use default filters ratio, i.e. assume that hat_filter=bar_filter")
  endif

  ! Register fields for output from this module.
  CS%diag => diag

  CS%id_u = register_diag_field('ocean_model', 'PG23_u', diag%axesCuL, Time, &
      'u velocity', 'm s-1', conversion=US%L_T_to_m_s)
  CS%id_v = register_diag_field('ocean_model', 'PG23_v', diag%axesCvL, Time, &
      'v velocity', 'm s-1', conversion=US%L_T_to_m_s)

  CS%id_uf = register_diag_field('ocean_model', 'PG23_uf', diag%axesCuL, Time, &
      'u filtered velocity', 'm s-1', conversion=US%L_T_to_m_s)
  CS%id_vf = register_diag_field('ocean_model', 'PG23_vf', diag%axesCvL, Time, &
      'v filtered velocity', 'm s-1', conversion=US%L_T_to_m_s)

  CS%id_vort_y = register_diag_field('ocean_model', 'PG23_vort_y', diag%axesCuL, Time, &
      'd/dy(vorticity)')
  CS%id_vort_yf = register_diag_field('ocean_model', 'PG23_vort_yf', diag%axesCuL, Time, &
      'd/dy(filtered vorticity)')
  CS%id_lap_vort_y = register_diag_field('ocean_model', 'PG23_lap_vort_y', diag%axesCuL, Time, &
      'd/dy(lap(vorticity))')
  CS%id_lap_vort_yf = register_diag_field('ocean_model', 'PG23_lap_vort_yf', diag%axesCuL, Time, &
      'd/dy(lap(filtered vorticity))')
  CS%id_smag_y = register_diag_field('ocean_model', 'PG23_smag_y', diag%axesCuL, Time, &
      'Zonal Vorticity flux of biharmonic Smagorinsky model')
  CS%id_smag_yf = register_diag_field('ocean_model', 'PG23_smag_yf', diag%axesCuL, Time, &
      'Zonal Vorticity flux of biharmonic Smagorinsky model on filtered level')
  CS%id_m_y = register_diag_field('ocean_model', 'PG23_m_y', diag%axesCuL, Time, &
      'Zonal Vorticity flux of eddy viscosity')
  CS%id_leo_y = register_diag_field('ocean_model', 'PG23_leo_y', diag%axesCuL, Time, &
      'Zonal Vorticity flux in Germano identity')
  CS%id_h_y = register_diag_field('ocean_model', 'PG23_h_y', diag%axesCuL, Time, &
      'Zonal Vorticity flux of SSM in Germano identity')
  CS%id_by = register_diag_field('ocean_model', 'PG23_by', diag%axesCuL, Time, &
      'Zonal Vorticity flux of Reynolds in Germano identity')
  CS%id_by_base = register_diag_field('ocean_model', 'PG23_by_base', diag%axesCuL, Time, &
      'Zonal Vorticity flux of Reynolds on base filter level')

  ! To DO: swap zonal and meridional in description
  CS%id_vort_x = register_diag_field('ocean_model', 'PG23_vort_x', diag%axesCvL, Time, &
      'd/dx(vorticity)')
  CS%id_vort_xf = register_diag_field('ocean_model', 'PG23_vort_xf', diag%axesCvL, Time, &
      'd/dx(filtered vorticity)')
  CS%id_lap_vort_x = register_diag_field('ocean_model', 'PG23_lap_vort_x', diag%axesCvL, Time, &
      'd/dx(lap(vorticity))')
  CS%id_lap_vort_xf = register_diag_field('ocean_model', 'PG23_lap_vort_xf', diag%axesCvL, Time, &
      'd/dx(lap(filtered vorticity))')
  CS%id_smag_x = register_diag_field('ocean_model', 'PG23_smag_x', diag%axesCvL, Time, &
      'Meridional Vorticity flux of biharmonic Smagorinsky model')
  CS%id_smag_xf = register_diag_field('ocean_model', 'PG23_smag_xf', diag%axesCvL, Time, &
      'Meridional Vorticity flux of biharmonic Smagorinsky model on filtered level')
  CS%id_m_x = register_diag_field('ocean_model', 'PG23_m_x', diag%axesCvL, Time, &
      'Meridional Vorticity flux of eddy viscosity')
  CS%id_leo_x = register_diag_field('ocean_model', 'PG23_leo_x', diag%axesCvL, Time, &
      'Meridional Vorticity flux in Germano identity')
  CS%id_h_x = register_diag_field('ocean_model', 'PG23_h_x', diag%axesCvL, Time, &
      'Meridional Vorticity flux of SSM in Germano identity')
  CS%id_bx = register_diag_field('ocean_model', 'PG23_bx', diag%axesCvL, Time, &
      'Meridional Vorticity flux of Reynolds in Germano identity')
  CS%id_bx_base = register_diag_field('ocean_model', 'PG23_bx_base', diag%axesCvL, Time, &
      'Meridional Vorticity flux of Reynolds on base filter level')

  CS%id_sh_xy = register_diag_field('ocean_model', 'PG23_sh_xy', diag%axesBL, Time, &
      'sh_xy')
  CS%id_sh_xyf = register_diag_field('ocean_model', 'PG23_sh_xyf', diag%axesBL, Time, &
      'filtered sh_xy')
  CS%id_vort_xy = register_diag_field('ocean_model', 'PG23_vort_xy', diag%axesBL, Time, &
      'vorticity')
  CS%id_vort_xyf = register_diag_field('ocean_model', 'PG23_vort_xyf', diag%axesBL, Time, &
      'filtered vorticity')
  CS%id_shear_mag = register_diag_field('ocean_model', 'PG23_shear_mag', diag%axesBL, Time, &
      'Shear magnitude')
  CS%id_shear_magf = register_diag_field('ocean_model', 'PG23_shear_magf', diag%axesBL, Time, &
      'Shear magnitude of filtered field')
  CS%id_lap_vort = register_diag_field('ocean_model', 'PG23_lap_vort', diag%axesBL, Time, &
      'Laplacian of vorticity')
  CS%id_lap_vortf = register_diag_field('ocean_model', 'PG23_lap_vortf', diag%axesBL, Time, &
      'Laplacian of filtered vorticity')

  CS%id_sh_xx = register_diag_field('ocean_model', 'PG23_sh_xx', diag%axesTL, Time, &
      'sh_xx')
  CS%id_sh_xxf = register_diag_field('ocean_model', 'PG23_sh_xxf', diag%axesTL, Time, &
      'filtered sh_xx')
  CS%id_lm = register_diag_field('ocean_model', 'PG23_lm', diag%axesTL, Time, &
      'Numerator of Germano identity')
  CS%id_mm = register_diag_field('ocean_model', 'PG23_mm', diag%axesTL, Time, &
      'Denominator of Germano identity')
  CS%id_lb = register_diag_field('ocean_model', 'PG23_lb', diag%axesTL, Time, &
      'Leonard * Backscatter in Germano identity')
  CS%id_mb = register_diag_field('ocean_model', 'PG23_mb', diag%axesTL, Time, &
      'Viscosity * Backscatter in Germano identity')
  CS%id_bb = register_diag_field('ocean_model', 'PG23_bb', diag%axesTL, Time, &
      'Backscatter * Backscatter in Germano identity')

  CS%id_smag = register_diag_field('ocean_model', 'smag_const', diag%axesZL, Time, &
      'Smagorinsky coefficient determined dynamically', 'nondim')
  CS%id_CR = register_diag_field('ocean_model', 'CR_const', diag%axesZL, Time, &
      'Reynolds coefficient determined dynamically', 'nondim')

  ! Clock IDs
  CS%id_clock_module = cpu_clock_id('(Ocean Perezhogin-Glazunov-2023)', grain=CLOCK_MODULE)
  CS%id_clock_mpi = cpu_clock_id('(PG23 MPI exchanges)', grain=CLOCK_ROUTINE, sync=.false.)
  CS%id_clock_filter = cpu_clock_id('(PG23 filter computations)', grain=CLOCK_ROUTINE, sync=.false.)
  CS%id_clock_post = cpu_clock_id('(PG23 post data)', grain=CLOCK_ROUTINE, sync=.false.)
  CS%id_clock_reduce = cpu_clock_id('(PG23 global reduce)', grain=CLOCK_ROUTINE, sync=.false.)

  allocate(CS%dx_dyT(SZI_(G),SZJ_(G)), source=0.)
  allocate(CS%dy_dxT(SZI_(G),SZJ_(G)), source=0.)
  allocate(CS%dx_dyBu(SZIB_(G),SZJB_(G)), source=0.)
  allocate(CS%dy_dxBu(SZIB_(G),SZJB_(G)), source=0.)
  allocate(CS%grid_sp_q4(SZIB_(G),SZJB_(G)), source=0.)

  do j=jsd,jed ; do i=isd,ied
    CS%dx_dyT(i,j) = G%dxT(i,j)*G%IdyT(i,j) ; CS%dy_dxT(i,j) = G%dyT(i,j)*G%IdxT(i,j)
  enddo ; enddo

  do J=JsdB,JedB ; do I=IsdB,IedB
    CS%dx_dyBu(I,J) = G%dxBu(I,J)*G%IdyBu(I,J) ; CS%dy_dxBu(I,J) = G%dyBu(I,J)*G%IdxBu(I,J)
    dx2q = G%dxBu(I,J)*G%dxBu(I,J) ; dy2q = G%dyBu(I,J)*G%dyBu(I,J)
    CS%grid_sp_q4(I,J) = ((2.0*dx2q*dy2q) / (dx2q+dy2q))**2
  enddo ; enddo

end subroutine PG23_init

!> Deallocate any variables allocated in PG23_init
subroutine PG23_end(CS)
  type(PG23_CS), intent(inout) :: CS  !< PG23 control structure.

  deallocate(CS%dx_dyT)
  deallocate(CS%dy_dxT)
  deallocate(CS%dx_dyBu)
  deallocate(CS%dy_dxBu)
  deallocate(CS%grid_sp_q4)

end subroutine PG23_end

!> This function estimates free parameters of PG23 closure 
!! using germano identity
!! In a simplest case of estimation of biharmonic Smagorinsky procedure:
!! * Exchange u,v velocities with full halo of 4 points
!! * Apply test filter to u, v
!! * Compute vorticity vort_xy and stress tensor sh_xx, sh_xy on both filter levels
!! * Apply test filter to u*vort_xy and v*vort_xy
!! * Compute leonard vorticity flux
!! * Compute modulus of strain rate |S| and grad(Lap(vort_xy)) on both filter levels
!! * Compute biharmonic Smagorinsky model of flux on two filters levels
!! * Apply test filter to the model on base level
!! * Compute alpha = Model_combined - hat(Model_base)
!! * Compute products, l*alpha and alpha*alpha
!! * Apply statistical averaging for products to infer the biharmonic Smagorinsky coefficient
subroutine PG23_germano_identity(u, v, h, smag_bi_const_DSM, C_R, leo_x, leo_y, bx_base, by_base, G, GV, CS)
  type(ocean_grid_type),         intent(in)    :: G  !< The ocean's grid structure.
  type(verticalGrid_type),       intent(in)    :: GV !< The ocean's vertical grid structure.
  type(PG23_CS),                 intent(inout) :: CS  !< PG23 control structure.

  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
        intent(in)  :: u  !< The zonal velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
        intent(in)  :: v  !< The meridional velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  &
        intent(in)     :: h  !< Layer thicknesses [H ~> m or kg m-2].
  real, dimension(SZK_(GV)), &
        intent(inout)  :: smag_bi_const_DSM, & !< The dynamically-estimated biharmonic Smagorinsky coefficient
                          C_R                  !< The dynamically-estimated backscatter coefficient
  real,  dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
        intent(inout) :: leo_x, bx_base !< Leonard vorticity x-flux 

  real,  dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
        intent(inout) :: leo_y, by_base !< Leonard vorticity y-flux 

  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)) ::  uf, & ! The fitered zonal velocity [L T-1 ~> m s-1]
                                        vort_y, & ! y derivative of Vertical vorticity
                                        vort_yf, & ! y derivative of filtered Vertical vorticity
                                        lap_vort_y, & ! y derivative of laplacian of Vertical vorticity
                                        lap_vort_yf, & ! y derivative of laplacian of filtered Vertical vorticity
                                        smag_y, & ! biharmonic Smagorinsky model on base level
                                        smag_yf, &  ! biharmonic Smagorinsky model on combined level
                                        m_y, & ! Smag. in dynamic model
                                        h_y, &     ! SSM y-flux in Germano identity
                                        by

  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)) ::  vf, & ! The fitered meridional velocity [L T-1 ~> m s-1]
                                        vort_x, & ! x derivative of Vertical vorticity
                                        vort_xf, & ! x derivative of filtered Vertical vorticity
                                        lap_vort_x, & ! x derivative of laplacian of Vertical vorticity
                                        lap_vort_xf, & ! x derivative of laplacian of filtered Vertical vorticity
                                        smag_x, & ! biharmonic Smagorinsky model on base level
                                        smag_xf, &   ! biharmonic Smagorinsky model on combined level
                                        m_x, & ! Smag. in dynamic model
                                        h_x, &    ! SSM x-flux in Germano identity
                                        bx

  real, dimension(SZIB_(G),SZJB_(G),SZK_(GV)) :: sh_xy, & ! horizontal shearing strain (du/dy + dv/dx)
                                        sh_xyf, & ! filtered horizontal shearing strain (du/dy + dv/dx)
                                        vort_xy, & ! vertical vorticity (dv/dx - du/dy)
                                        vort_xyf, & ! filtered vertical vorticity (dv/dx - du/dy)
                                        shear_mag, & ! Magnitude of shear in q points [T-1 ~> s-1]
                                        shear_magf, & ! Magnitude of the filtered shear in q points [T-1 ~> s-1]
                                        lap_vort, & ! Laplacian of Vertical vorticity
                                        lap_vortf  ! Laplacian of filtered Vertical vorticity

  real, dimension(SZI_(G),SZJ_(G),SZK_(GV))   :: sh_xx, & ! horizontal tension (du/dx - dv/dy)
                                        sh_xxf, &  ! filtered horizontal tension (du/dx - dv/dy)
                                        lm, &      ! numerator of Germano identity
                                        mm, &         ! denominator of Germano identity
                                        lb, mb, bb

  integer :: k, nz 
  real, dimension(SZK_(GV)) :: lm_sum, mm_sum, lb_sum, mb_sum, bb_sum
  real :: dummy_argument

  !type(group_pass_type) :: pass_uv  ! A handle used for group halo passes

  call cpu_clock_begin(CS%id_clock_module)

  ! Note: halo without this exchange is only 3 points, which is enough for
  ! Domain-averaged biharmonic Smagorinsky model
  !call create_group_pass(pass_uv, u, v, G%Domain)
  !call do_group_pass(pass_uv, G%domain, clock=CS%id_clock_mpi)

  nz = GV%ke

  uf = 0
  vf = 0
  vort_xyf = 0
  
  do k=1,nz
    ! Now these arrays have halo 4
    uf(:,:,k) = u(:,:,k)
    vf(:,:,k) = v(:,:,k)
    
    ! Apply test filter to velocities (halo=3 is default in MOM6 for velocities)
    call filter_wrapper(G, GV, CS, CS%test_width, halo=3, niter=1, u=uf(:,:,k), v=vf(:,:,k))

    ! Compute velocity gradients for eddy viscosity model on base level
    ! We impose actual halo for filtered and unfiltered velocities
    call compute_velocity_gradients(u(:,:,k), v(:,:,k), &
      sh_xx(:,:,k), sh_xy(:,:,k), vort_xy(:,:,k), shear_mag(:,:,k), G, GV, CS, halo=3)
    ! On combined level
    call compute_velocity_gradients(uf(:,:,k), vf(:,:,k), &
      sh_xxf(:,:,k), sh_xyf(:,:,k), vort_xyf(:,:,k), shear_magf(:,:,k), G, GV, CS, halo=2)

    ! Compute grad(Lap(vorticity)) for eddy viscosity model
    ! In both cases available halo of vorticity is 1 point smaller than for velocity
    call compute_vorticity_gradients(vort_xy(:,:,k), &
      vort_x(:,:,k), vort_y(:,:,k), lap_vort(:,:,k), lap_vort_x(:,:,k), lap_vort_y(:,:,k), G, GV, CS, halo=2)
    ! On combined level
    call compute_vorticity_gradients(vort_xyf(:,:,k), &
      vort_xf(:,:,k), vort_yf(:,:,k), lap_vortf(:,:,k), lap_vort_xf(:,:,k), lap_vort_yf(:,:,k), G, GV, CS, halo=1)

    if (CS%zelong_dynamic) then
      call  biharmonic_Smagorinsky(lap_vort_xf(:,:,k), lap_vort_yf(:,:,k), shear_magf(:,:,k), &
        smag_xf(:,:,k), smag_yf(:,:,k), G, GV, CS, halo=0, scaling_coefficient=1.0)
      m_x(:,:,k) = smag_xf(:,:,k)
      m_y(:,:,k) = smag_yf(:,:,k)
    else
      ! Computation of biharmonic Smagorinsky model of vorticity flux
      ! Halo is 1 point smaller than for the vorticity
      call  biharmonic_Smagorinsky(lap_vort_x(:,:,k), lap_vort_y(:,:,k), shear_mag(:,:,k), &
      smag_x(:,:,k), smag_y(:,:,k), G, GV, CS, halo=1, scaling_coefficient=1.0)
      ! Note: filters ratio is in 4th power because model is biharmonic
      call  biharmonic_Smagorinsky(lap_vort_xf(:,:,k), lap_vort_yf(:,:,k), shear_magf(:,:,k), &
      smag_xf(:,:,k), smag_yf(:,:,k), G, GV, CS, halo=0, scaling_coefficient=CS%filters_ratio**4)

      ! Filter Smagorinky model on base level
      call filter_wrapper(G, GV, CS, CS%test_width, halo=1, niter=1, u=smag_y(:,:,k), v=smag_x(:,:,k))

      ! Eddy viscosity in Germano identity with halo 0 (halo 1 is possible if needed)
      m_x(:,:,k) = smag_xf(:,:,k) - smag_x(:,:,k)
      m_y(:,:,k) = smag_yf(:,:,k) - smag_y(:,:,k)
    endif

    ! leo_x = bar(u * vort_xy) - bar(u) * bar(vort_xy)
    ! leo_y = bar(v * vort_xy) - bar(v) * bar(vort_xy)
    ! Output Leonard flux has halo 0 (halo 2 is also possible if needed)
    call compute_leonard_flux(leo_x(:,:,k), leo_y(:,:,k), h_x(:,:,k), h_y(:,:,k),                        &
                              bx(:,:,k), by(:,:,k), bx_base(:,:,k), by_base(:,:,k),                      &
                              u(:,:,k), v(:,:,k), vort_xy(:,:,k), uf(:,:,k), vf(:,:,k), vort_xyf(:,:,k), &
                              G, GV, CS, CS%test_width, halo=1)
    ! Output halo is 0; 
    ! So, dynamic biharmonic Smagorinsky model requires halo 3 (as default for velocities)
    if (CS%ssm) then
      if (CS%reynolds) then
        call compute_lm_mm(leo_x(:,:,k) - h_x(:,:,k), leo_y(:,:,k) - h_y(:,:,k), m_x(:,:,k), m_y(:,:,k), lm(:,:,k), mm(:,:,k), G, GV, CS, halo=1, &
                           bx=bx(:,:,k), by=by(:,:,k), lb=lb(:,:,k), mb=mb(:,:,k), bb=bb(:,:,k))
      else
        call compute_lm_mm(leo_x(:,:,k) - h_x(:,:,k), leo_y(:,:,k) - h_y(:,:,k), m_x(:,:,k), m_y(:,:,k), lm(:,:,k), mm(:,:,k), G, GV, CS, halo=1)
      endif
    else
      call compute_lm_mm(leo_x(:,:,k), leo_y(:,:,k), m_x(:,:,k), m_y(:,:,k), lm(:,:,k), mm(:,:,k), G, GV, CS, halo=1)
    endif
    
  enddo ! end of k loop

  call cpu_clock_begin(CS%id_clock_reduce)
  
  if (CS%reduce==0) then
    dummy_argument = reproducing_sum(lm(G%isc:G%iec,G%jsc:G%jec,1:nz), sums=lm_sum)
  else if (CS%reduce==1) then
    dummy_argument = reproducing_sum(max(lm(G%isc:G%iec,G%jsc:G%jec,1:nz),0.0), sums=lm_sum)
  endif
  dummy_argument = reproducing_sum(mm(G%isc:G%iec,G%jsc:G%jec,1:nz), sums=mm_sum)
  if (CS%reynolds) then
    dummy_argument = reproducing_sum(lb(G%isc:G%iec,G%jsc:G%jec,1:nz), sums=lb_sum)
    dummy_argument = reproducing_sum(bb(G%isc:G%iec,G%jsc:G%jec,1:nz), sums=bb_sum)
    dummy_argument = reproducing_sum(mb(G%isc:G%iec,G%jsc:G%jec,1:nz), sums=mb_sum)
  endif

  call cpu_clock_end(CS%id_clock_reduce)

  smag_bi_const_DSM = max(lm_sum / (mm_sum + 1e-40), 0.0)

  if (CS%reynolds) then
    C_R = max((lb_sum - smag_bi_const_DSM * mb_sum) / (bb_sum + 1e-40), 0.0)
  endif

  call cpu_clock_begin(CS%id_clock_post)

  if (CS%id_u>0)             call post_data(CS%id_u, u, CS%diag)
  if (CS%id_v>0)             call post_data(CS%id_v, v, CS%diag)
  if (CS%id_uf>0)            call post_data(CS%id_uf, uf, CS%diag)
  if (CS%id_vf>0)            call post_data(CS%id_vf, vf, CS%diag)

  if (CS%id_vort_y>0)        call post_data(CS%id_vort_y, vort_y, CS%diag)
  if (CS%id_vort_yf>0)       call post_data(CS%id_vort_yf, vort_yf, CS%diag)
  if (CS%id_lap_vort_y>0)    call post_data(CS%id_lap_vort_y, lap_vort_y, CS%diag)
  if (CS%id_lap_vort_yf>0)   call post_data(CS%id_lap_vort_yf, lap_vort_yf, CS%diag)
  if (CS%id_smag_y>0)        call post_data(CS%id_smag_y, smag_y, CS%diag)
  if (CS%id_smag_yf>0)       call post_data(CS%id_smag_yf, smag_yf, CS%diag)
  if (CS%id_m_y>0)           call post_data(CS%id_m_y, m_y, CS%diag)
  if (CS%id_leo_y>0)         call post_data(CS%id_leo_y, leo_y, CS%diag)
  if (CS%id_h_y>0)           call post_data(CS%id_h_y, h_y, CS%diag)
 
  if (CS%id_vort_x>0)        call post_data(CS%id_vort_x, vort_x, CS%diag)
  if (CS%id_vort_xf>0)       call post_data(CS%id_vort_xf, vort_xf, CS%diag)
  if (CS%id_lap_vort_x>0)    call post_data(CS%id_lap_vort_x, lap_vort_x, CS%diag)
  if (CS%id_lap_vort_xf>0)   call post_data(CS%id_lap_vort_xf, lap_vort_xf, CS%diag)
  if (CS%id_smag_x>0)        call post_data(CS%id_smag_x, smag_x, CS%diag)
  if (CS%id_smag_xf>0)       call post_data(CS%id_smag_xf, smag_xf, CS%diag)
  if (CS%id_m_x>0)           call post_data(CS%id_m_x, m_x, CS%diag)
  if (CS%id_leo_x>0)         call post_data(CS%id_leo_x, leo_x, CS%diag)
  if (CS%id_h_x>0)           call post_data(CS%id_h_x, h_x, CS%diag)

  if (CS%id_sh_xy>0)         call post_data(CS%id_sh_xy, sh_xy, CS%diag)
  if (CS%id_sh_xyf>0)        call post_data(CS%id_sh_xyf, sh_xyf, CS%diag)
  if (CS%id_vort_xy>0)       call post_data(CS%id_vort_xy, vort_xy, CS%diag)
  if (CS%id_vort_xyf>0)      call post_data(CS%id_vort_xyf, vort_xyf, CS%diag)
  if (CS%id_shear_mag>0)     call post_data(CS%id_shear_mag, shear_mag, CS%diag)
  if (CS%id_shear_magf>0)    call post_data(CS%id_shear_magf, shear_magf, CS%diag)
  if (CS%id_lap_vort>0)      call post_data(CS%id_lap_vort, lap_vort, CS%diag)
  if (CS%id_lap_vortf>0)     call post_data(CS%id_lap_vortf, lap_vortf, CS%diag)

  if (CS%id_sh_xx>0)         call post_data(CS%id_sh_xx, sh_xx, CS%diag)
  if (CS%id_sh_xxf>0)        call post_data(CS%id_sh_xxf, sh_xxf, CS%diag)
  if (CS%id_lm>0)            call post_data(CS%id_lm, lm, CS%diag)
  if (CS%id_mm>0)            call post_data(CS%id_mm, mm, CS%diag)
  if (CS%id_bb>0)            call post_data(CS%id_bb, bb, CS%diag)
  if (CS%id_mb>0)            call post_data(CS%id_mb, mb, CS%diag)
  if (CS%id_lb>0)            call post_data(CS%id_lb, lb, CS%diag)

  if (CS%id_bx>0)            call post_data(CS%id_bx, bx, CS%diag)
  if (CS%id_bx_base>0)       call post_data(CS%id_bx_base, bx_base, CS%diag)

  if (CS%id_by>0)            call post_data(CS%id_by, by, CS%diag)
  if (CS%id_by_base>0)       call post_data(CS%id_by_base, by_base, CS%diag)

  if (CS%id_smag>0)           call post_data(CS%id_smag, smag_bi_const_DSM, CS%diag)
  if (CS%id_CR>0)             call post_data(CS%id_CR, C_R, CS%diag)

  call cpu_clock_end(CS%id_clock_post)

  call cpu_clock_end(CS%id_clock_module)

  if (CS%offline) then
    smag_bi_const_DSM = 0.06
    leo_x = 0.
    leo_y = 0.
  endif

end subroutine PG23_germano_identity

!> This function computes scalar product of leonard flux
!! and eddy viscosity model in center points
subroutine compute_lm_mm(leo_x, leo_y, m_x, m_y, lm, mm, &
                         G, GV, CS, halo,                &
                         bx, by, lb, mb, bb)
  type(ocean_grid_type),   intent(in) :: G       !< The ocean's grid structure.
  type(verticalGrid_type), intent(in) :: GV      !< The ocean's vertical grid structure
  type(PG23_CS),           intent(in) :: CS      !< PG23 control structure.
  integer, intent(in) :: halo !< Currently available halo points for vorticity fluxes
                              !! in symmetric memory model
  
  real, dimension(SZI_(G),SZJB_(G)), intent(in) :: leo_x, & !< Leonard vorticity x-flux
                                                   m_x      !< Eddy visocitty in Germano identity x-flux  
  real, dimension(SZIB_(G),SZJ_(G)), intent(in) :: leo_y, & !< Leonard vorticity y-flux
                                                   m_y      !< Eddy visocitty in Germano identity y-flux  
  real, dimension(SZI_(G),SZJB_(G)), optional, intent(in) :: bx !< Reynolds flux
  real, dimension(SZIB_(G),SZJ_(G)), optional, intent(in) :: by !< Reynolds flux

  real, dimension(SZI_(G),SZJ_(G)), intent(inout) :: lm, &  !< Germano identity: Leonard flux times eddy viscosity flux
                                                     mm     !< Germano identity: eddy viscosity flux squared
  real, dimension(SZI_(G),SZJ_(G)), optional, intent(inout) :: &
                                                     lb, mb, bb
  
  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq
  integer :: i, j

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  ! This operation does not lose halo
  do j=js-halo,je+halo ; do i=is-halo, ie+halo
    ! Here we use two-point interpolation, so coefficient is 0.5
    lm(i,j) = 0.5 * ((leo_x(i,J) * m_x(i,J) + leo_x(i,J-1) * m_x(i,J-1)) + &
                     (leo_y(I,j) * m_y(I,j) + leo_y(I-1,j) * m_y(I-1,j))) * G%mask2dT(i,j) * G%areaT(i,j)

    mm(i,j) = 0.5 * ((m_x(i,J) * m_x(i,J) + m_x(i,J-1) * m_x(i,J-1)) + &
                     (m_y(I,j) * m_y(I,j) + m_y(I-1,j) * m_y(I-1,j))) * G%mask2dT(i,j) * G%areaT(i,j)
    if (present(bx)) then
      lb(i,j) = 0.5 * ((leo_x(i,J) * bx(i,J) + leo_x(i,J-1) * bx(i,J-1)) + &
                       (leo_y(I,j) * by(I,j) + leo_y(I-1,j) * by(I-1,j))) * G%mask2dT(i,j) * G%areaT(i,j)
      mb(i,j) = 0.5 * ((m_x(i,J) * bx(i,J) + m_x(i,J-1) * bx(i,J-1)) + &
                       (m_y(I,j) * by(I,j) + m_y(I-1,j) * by(I-1,j))) * G%mask2dT(i,j) * G%areaT(i,j)
      bb(i,j) = 0.5 * ((bx(i,J) * bx(i,J) + bx(i,J-1) * bx(i,J-1)) + &
                       (by(I,j) * by(I,j) + by(I-1,j) * by(I-1,j))) * G%mask2dT(i,j) * G%areaT(i,j)
    endif
  enddo ; enddo
  
end subroutine compute_lm_mm

!> This function computes the leonard flux
!! leo_x = bar(u * vort_xy) - bar(u) * bar(vort_xy)
!! leo_y = bar(v * vort_xy) - bar(v) * bar(vort_xy)
subroutine compute_leonard_flux(leo_x, leo_y, h_x, h_y,            &
                                  bx, by, bx_base, by_base,        &
                                  u, v, vort_xy, uf, vf, vort_xyf, &
                                  G, GV, CS, filter_width, halo)
  type(ocean_grid_type),   intent(in) :: G       !< The ocean's grid structure.
  type(verticalGrid_type), intent(in) :: GV      !< The ocean's vertical grid structure
  type(PG23_CS),           intent(in) :: CS      !< PG23 control structure.
  integer, intent(in) :: halo !< Currently available halo points for vorticity (vort_xy, not vort_xyf)
                              !! (velocity assumed to have larger halo) in symmetric memory model
  real, intent(in) :: filter_width !< Filter width (nondim) used to compute bar(u*vort_xy)
                                      !! and used to compute bar(u), var(v) and bar(vort_xy) before filter call

  real, dimension(SZIB_(G),SZJ_(G)), &
           intent(in) :: u, & !< The zonal velocity [L T-1 ~> m s-1].
                         uf   !< The filtered zonal velocity [L T-1 ~> m s-1].
            
  real, dimension(SZI_(G),SZJB_(G)), &
           intent(in) :: v, & !< The meridional velocity [L T-1 ~> m s-1].
                         vf   !< The filtered meridional velocity [L T-1 ~> m s-1].

  real, dimension(SZIB_(G),SZJB_(G)), &
      intent(in)    :: vort_xy, &     !< Vertical vorticity (dv/dx - du/dy) [T-1 ~> s-1]
                       vort_xyf       !< Filtered vertical vorticity (dv/dx - du/dy) [T-1 ~> s-1]

  real, dimension(SZI_(G),SZJB_(G)), intent(inout) :: leo_x ! Leonard vorticity x-flux
  real, dimension(SZIB_(G),SZJ_(G)), intent(inout) :: leo_y ! Leonard vorticity y-flux

  real, dimension(SZI_(G),SZJB_(G)), intent(inout) :: h_x ! SSM x-flux contribution in Germano identity
  real, dimension(SZIB_(G),SZJ_(G)), intent(inout) :: h_y ! SSM y-flux contribution in Germano identity

  real, dimension(SZI_(G),SZJB_(G)), intent(inout)  :: bx
  real, dimension(SZIB_(G),SZJ_(G)), intent(inout)  :: by

  real, dimension(SZI_(G),SZJB_(G)), intent(inout)  :: bx_base ! Reynolds on base filter level
  real, dimension(SZIB_(G),SZJ_(G)), intent(inout)  :: by_base ! Reynolds on base filter level

  real, dimension(SZI_(G),SZJB_(G)) :: u_hat_vort_hat ! hat(u) * hat(vort_xy)
  real, dimension(SZIB_(G),SZJ_(G)) :: v_hat_vort_hat ! hat(v) * hat(vort_xy)
  real, dimension(SZI_(G),SZJB_(G)) :: leo_xf ! Leonard vorticity x-flux filtered
  real, dimension(SZIB_(G),SZJ_(G)) :: leo_yf ! Leonard vorticity y-flux filtered

  real, dimension(SZIB_(G),SZJ_(G)) :: ur_base ! Residual velocity on base filter level
  real, dimension(SZI_(G),SZJB_(G)) :: vr_base ! Residual velocity on base filter level
  real, dimension(SZIB_(G),SZJB_(G)) :: vortr_base ! Residual vorticity on base filter level

  real, dimension(SZI_(G),SZJB_(G)) :: bx_basef ! Reynolds on base filter level filtered
  real, dimension(SZIB_(G),SZJ_(G)) :: by_basef ! Reynolds on base filter level filtered


  real, dimension(SZIB_(G),SZJ_(G)) :: ur_comb
  real, dimension(SZI_(G),SZJB_(G)) :: vr_comb
  real, dimension(SZIB_(G),SZJB_(G)) :: vortr_comb

  real, dimension(SZIB_(G),SZJ_(G))  :: uff
  real, dimension(SZI_(G),SZJB_(G))  :: vff
  real, dimension(SZIB_(G),SZJB_(G)) :: vort_xyff
    
  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq
  integer :: i, j

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  uff = 0
  vff = 0
  vort_xyff = 0
  leo_xf = 0
  leo_yf = 0
  u_hat_vort_hat = 0
  v_hat_vort_hat = 0
  h_x = 0
  h_y = 0
  leo_x = 0
  leo_y = 0

  ! u * vort_xy
  do J=Jsq-halo,Jeq+halo ; do i=is-halo,ie+halo
    ! Here 0.125 is interpolation weight, i.e. 1/4 * 1/2 = 1/8 = 0.125
    ! Similarly to computing vorticity gradients, halo is not reduced
    leo_x(i,J) = 0.125 * ((u(I,j) + u(I-1,j+1)) + (u(I-1,j) + u(I,j+1))) * (vort_xy(I,J) + vort_xy(I-1,J)) * G%mask2dCv(i,J)
  enddo ; enddo

  ! v * vort_xy
  do j=js-halo,je+halo ; do I=Isq-halo,Ieq+halo
    ! Here 0.125 is interpolation weight, i.e. 1/4 * 1/2 = 1/8 = 0.125
    leo_y(I,j) = 0.125 * ((v(i,J) + v(i+1,J-1)) + (v(i,J-1) + v(i+1,J))) * (vort_xy(I,J) + vort_xy(I,J-1)) * G%mask2dCu(I,j)
  enddo ; enddo

  ! bar(u * vort_xy) and bar(v * vort_xy)
  call filter_wrapper(G, GV, CS, filter_width, halo=halo, niter=1, u=leo_y, v=leo_x)
  
  ! leo_x = bar(u * vort_xy) - bar(u) * bar(vort_xy)
  do J=Jsq-(halo-1),Jeq+(halo-1) ; do i=is-(halo-1),ie+(halo-1)
    u_hat_vort_hat(i,J) = 0.125 * ((uf(I,j) + uf(I-1,j+1)) + (uf(I-1,j) + uf(I,j+1))) * (vort_xyf(I,J) + vort_xyf(I-1,J)) * G%mask2dCv(i,J)
  enddo ; enddo

  ! leo_y = bar(v * vort_xy) - bar(v) * bar(vort_xy)
  do j=js-(halo-1),je+(halo-1) ; do I=Isq-(halo-1),Ieq+(halo-1)
    v_hat_vort_hat(I,j) = 0.125 * ((vf(i,J) + vf(i+1,J-1)) + (vf(i,J-1) + vf(i+1,J))) * (vort_xyf(I,J) + vort_xyf(I,J-1)) * G%mask2dCu(I,j)
  enddo ; enddo

  ! leo_x = bar(u * vort_xy) - bar(u) * bar(vort_xy)
  do J=Jsq-(halo-1),Jeq+(halo-1) ; do i=is-(halo-1),ie+(halo-1)
    leo_x(i,J) = leo_x(i,J) - u_hat_vort_hat(i,J)
  enddo ; enddo

  ! leo_y = bar(v * vort_xy) - bar(v) * bar(vort_xy)
  do j=js-(halo-1),je+(halo-1) ; do I=Isq-(halo-1),Ieq+(halo-1)
    leo_y(I,j) = leo_y(I,j) - v_hat_vort_hat(I,j)
  enddo ; enddo

  if (CS%ssm) then
    call pass_vector(v_hat_vort_hat, u_hat_vort_hat, G%Domain, clock=CS%id_clock_mpi)
    uff = uf
    vff = vf
    vort_xyff = vort_xyf
    call pass_vector(uff, vff, G%Domain, clock=CS%id_clock_mpi)
    call pass_var(vort_xyff, G%Domain, position=CORNER, clock=CS%id_clock_mpi)

    if (CS%zelong_dynamic) then
      call filter_wrapper(G, GV, CS, filter_width, halo=4, niter=1, u=v_hat_vort_hat, v=u_hat_vort_hat)
      call filter_wrapper(G, GV, CS, filter_width, halo=4, niter=1, u=uff, v=vff, q=vort_xyff)
    else
      call pass_vector(leo_y, leo_x, G%Domain, clock=CS%id_clock_mpi) ! Because leo_x is in v points

      leo_xf = leo_x
      leo_yf = leo_y

      call filter_wrapper(G, GV, CS, filter_width, halo=4, niter=1, u=leo_yf, v=leo_xf)
      call filter_wrapper(G, GV, CS, filter_width, halo=4, niter=2, u=v_hat_vort_hat, v=u_hat_vort_hat)
      call filter_wrapper(G, GV, CS, filter_width, halo=4, niter=2, u=uff, v=vff, q=vort_xyff)
    endif

    do J=Jsq-1,Jeq+1 ; do i=is-1,ie+1
      h_x(i,J) = u_hat_vort_hat(i,J) - &
        0.125 * ((uff(I,j) + uff(I-1,j+1)) + (uff(I-1,j) + uff(I,j+1))) * (vort_xyff(I,J) + vort_xyff(I-1,J)) * G%mask2dCv(i,J)
      if (not(CS%zelong_dynamic)) then
        h_x(i,J) = h_x(i,J) - leo_xf(i,J)
      endif
    enddo ; enddo

    do j=js-1,je+1 ; do I=Isq-1,Ieq+1
      h_y(I,j) = v_hat_vort_hat(I,j) - &
        0.125 * ((vff(i,J) + vff(i+1,J-1)) + (vff(i,J-1) + vff(i+1,J))) * (vort_xyff(I,J) + vort_xyff(I,J-1)) * G%mask2dCu(I,j)
      if (not(CS%zelong_dynamic)) then
        h_y(I,j) = h_y(I,j) - leo_yf(I,j)
      endif
    enddo ; enddo
  endif

  if (CS%reynolds) then
    if (.not.CS%ssm) then
      call MOM_error(FATAL, &
         "MOM_dynamic_closures: Not implemented")
    endif

    if (not(CS%zelong_dynamic)) then

      ur_base = u - uf
      vr_base = v - vf
      vortr_base = vort_xy - vort_xyf

      do J=Jsq-2,Jeq+2 ; do i=is-2,ie+2
        bx_base(i,J) = 0.125 * ((ur_base(I,j) + ur_base(I-1,j+1)) + (ur_base(I-1,j) + ur_base(I,j+1))) * (vortr_base(I,J) + vortr_base(I-1,J)) * G%mask2dCv(i,J)
      enddo ; enddo

      do j=js-2,je+2 ; do I=Isq-2,Ieq+2
        by_base(I,j) = 0.125 * ((vr_base(i,J) + vr_base(i+1,J-1)) + (vr_base(i,J-1) + vr_base(i+1,J))) * (vortr_base(I,J) + vortr_base(I,J-1)) * G%mask2dCu(I,j)
      enddo ; enddo

      call filter_wrapper(G, GV, CS, filter_width, halo=4, niter=1, u=by_base, v=bx_base)
      call filter_wrapper(G, GV, CS, filter_width, halo=4, niter=1, u=ur_base, v=vr_base, q=vortr_base)

      do J=Jsq-2,Jeq+2 ; do i=is-2,ie+2
        bx_base(i,J) = bx_base(i,J) -  0.125 * ((ur_base(I,j) + ur_base(I-1,j+1)) + (ur_base(I-1,j) + ur_base(I,j+1))) * (vortr_base(I,J) + vortr_base(I-1,J)) * G%mask2dCv(i,J)
      enddo ; enddo

      do j=js-2,je+2 ; do I=Isq-2,Ieq+2
        by_base(I,j) = by_base(I,j) - 0.125 * ((vr_base(i,J) + vr_base(i+1,J-1)) + (vr_base(i,J-1) + vr_base(i+1,J))) * (vortr_base(I,J) + vortr_base(I,J-1)) * G%mask2dCu(I,j)
      enddo ; enddo

      call pass_vector(by_base, bx_base, G%Domain, clock=CS%id_clock_mpi)
      bx_basef = bx_base
      by_basef = by_base

      call filter_wrapper(G, GV, CS, filter_width, halo=4, niter=1, u=by_basef, v=bx_basef)

      !!!!!!!!!!! Combined level !!!!!!!!!!!!!
      ur_comb = uf - uff
      vr_comb = vf - vff
      vortr_comb = vort_xyf - vort_xyff

      do J=Jsq-2,Jeq+2 ; do i=is-2,ie+2
        bx(i,J) = 0.125 * ((ur_comb(I,j) + ur_comb(I-1,j+1)) + (ur_comb(I-1,j) + ur_comb(I,j+1))) * (vortr_comb(I,J) + vortr_comb(I-1,J)) * G%mask2dCv(i,J)
      enddo ; enddo

      do j=js-2,je+2 ; do I=Isq-2,Ieq+2
        by(I,j) = 0.125 * ((vr_comb(i,J) + vr_comb(i+1,J-1)) + (vr_comb(i,J-1) + vr_comb(i+1,J))) * (vortr_comb(I,J) + vortr_comb(I,J-1)) * G%mask2dCu(I,j)
      enddo ; enddo

      call pass_vector(ur_comb, vr_comb, G%Domain, clock=CS%id_clock_mpi)
      call pass_var(vortr_comb, G%Domain, position=CORNER, clock=CS%id_clock_mpi)
      call pass_vector(by, bx, G%Domain, clock=CS%id_clock_mpi)

      call filter_wrapper(G, GV, CS, filter_width, halo=4, niter=2, u=by, v=bx)
      call filter_wrapper(G, GV, CS, filter_width, halo=4, niter=2, u=ur_comb, v=vr_comb, q=vortr_comb)

      ! Combined level
      do J=Jsq-2,Jeq+2 ; do i=is-2,ie+2
        bx(i,J) = bx(i,J) - 0.125 * ((ur_comb(I,j) + ur_comb(I-1,j+1)) + (ur_comb(I-1,j) + ur_comb(I,j+1))) * (vortr_comb(I,J) + vortr_comb(I-1,J)) * G%mask2dCv(i,J)
      enddo ; enddo

      do j=js-2,je+2 ; do I=Isq-2,Ieq+2
        by(I,j) = by(I,j) - 0.125 * ((vr_comb(i,J) + vr_comb(i+1,J-1)) + (vr_comb(i,J-1) + vr_comb(i+1,J))) * (vortr_comb(I,J) + vortr_comb(I,J-1)) * G%mask2dCu(I,j)
      enddo ; enddo

      ! Germano identity level
      bx = bx - bx_basef
      by = by - by_basef
    else

      ur_base = u - uf
      vr_base = v - vf
      vortr_base = vort_xy - vort_xyf

      do J=Jsq-2,Jeq+2 ; do i=is-2,ie+2
        bx_base(i,J) = 0.125 * ((ur_base(I,j) + ur_base(I-1,j+1)) + (ur_base(I-1,j) + ur_base(I,j+1))) * (vortr_base(I,J) + vortr_base(I-1,J)) * G%mask2dCv(i,J)
      enddo ; enddo

      do j=js-2,je+2 ; do I=Isq-2,Ieq+2
        by_base(I,j) = 0.125 * ((vr_base(i,J) + vr_base(i+1,J-1)) + (vr_base(i,J-1) + vr_base(i+1,J))) * (vortr_base(I,J) + vortr_base(I,J-1)) * G%mask2dCu(I,j)
      enddo ; enddo

      call filter_wrapper(G, GV, CS, filter_width, halo=4, niter=1, u=by_base, v=bx_base)
      call filter_wrapper(G, GV, CS, filter_width, halo=4, niter=1, u=ur_base, v=vr_base, q=vortr_base)
 
      do J=Jsq-2,Jeq+2 ; do i=is-2,ie+2
        bx_base(i,J) = bx_base(i,J) -  0.125 * ((ur_base(I,j) + ur_base(I-1,j+1)) + (ur_base(I-1,j) + ur_base(I,j+1))) * (vortr_base(I,J) + vortr_base(I-1,J)) * G%mask2dCv(i,J)
      enddo ; enddo

      do j=js-2,je+2 ; do I=Isq-2,Ieq+2
        by_base(I,j) = by_base(I,j) - 0.125 * ((vr_base(i,J) + vr_base(i+1,J-1)) + (vr_base(i,J-1) + vr_base(i+1,J))) * (vortr_base(I,J) + vortr_base(I,J-1)) * G%mask2dCu(I,j)
      enddo ; enddo

      call pass_vector(by_base, bx_base, G%Domain, clock=CS%id_clock_mpi)

      !!!!!!!!!!! Combined level !!!!!!!!!!!!!
      ur_comb = uf - uff
      vr_comb = vf - vff
      vortr_comb = vort_xyf - vort_xyff

      do J=Jsq-2,Jeq+2 ; do i=is-2,ie+2
        bx(i,J) = 0.125 * ((ur_comb(I,j) + ur_comb(I-1,j+1)) + (ur_comb(I-1,j) + ur_comb(I,j+1))) * (vortr_comb(I,J) + vortr_comb(I-1,J)) * G%mask2dCv(i,J)
      enddo ; enddo

      do j=js-2,je+2 ; do I=Isq-2,Ieq+2
        by(I,j) = 0.125 * ((vr_comb(i,J) + vr_comb(i+1,J-1)) + (vr_comb(i,J-1) + vr_comb(i+1,J))) * (vortr_comb(I,J) + vortr_comb(I,J-1)) * G%mask2dCu(I,j)
      enddo ; enddo

      call pass_vector(ur_comb, vr_comb, G%Domain, clock=CS%id_clock_mpi)
      call pass_var(vortr_comb, G%Domain, position=CORNER, clock=CS%id_clock_mpi)
      call pass_vector(by, bx, G%Domain, clock=CS%id_clock_mpi)

      call filter_wrapper(G, GV, CS, filter_width, halo=4, niter=1, u=by, v=bx)
      call filter_wrapper(G, GV, CS, filter_width, halo=4, niter=1, u=ur_comb, v=vr_comb, q=vortr_comb)

      ! Combined level
      do J=Jsq-2,Jeq+2 ; do i=is-2,ie+2
        bx(i,J) = bx(i,J) - 0.125 * ((ur_comb(I,j) + ur_comb(I-1,j+1)) + (ur_comb(I-1,j) + ur_comb(I,j+1))) * (vortr_comb(I,J) + vortr_comb(I-1,J)) * G%mask2dCv(i,J)
      enddo ; enddo

      do j=js-2,je+2 ; do I=Isq-2,Ieq+2
        by(I,j) = by(I,j) - 0.125 * ((vr_comb(i,J) + vr_comb(i+1,J-1)) + (vr_comb(i,J-1) + vr_comb(i+1,J))) * (vortr_comb(I,J) + vortr_comb(I,J-1)) * G%mask2dCu(I,j)
      enddo ; enddo

    endif
    
  endif
  
end subroutine compute_leonard_flux

!> This function computes the biharmonic Smagorinsky model of 
!! vorticity flux:
!! bar(u'*vort_xy') = (dx)^4 * |S| * d/dx(Lap(vort_xy))
!! bar(v'*vort_xy') = (dx)^4 * |S| * d/dy(Lap(vort_xy))
subroutine biharmonic_Smagorinsky(lap_vort_x, lap_vort_y, shear_mag, &
                                  smag_x, smag_y, G, GV, CS, halo, scaling_coefficient)
  type(ocean_grid_type),   intent(in) :: G       !< The ocean's grid structure.
  type(verticalGrid_type), intent(in) :: GV      !< The ocean's vertical grid structure
  type(PG23_CS),           intent(in) :: CS      !< PG23 control structure.
  integer, intent(in) :: halo !< Currently available halo points for lap_vort_x, lap_vort_y
                              !! in symmetric memory model
  real, intent(in) :: scaling_coefficient !< The Smagorinsky coefficient or filters ratio

  real, dimension(SZI_(G),SZJB_(G)), &
      intent(in) :: lap_vort_x  !< x derivative of Laplacian of Vertical vorticity (dv/dx - du/dy) [T-1 L-2 ~> s-1 m-2]
  real, dimension(SZIB_(G),SZJ_(G)), &
      intent(in) :: lap_vort_y  !< y derivative of Laplacian of Vertical vorticity (dv/dx - du/dy) [T-1 L-2 ~> s-1 m-2]
  real, dimension(SZIB_(G),SZJB_(G)), &
      intent(in) :: shear_mag   !< Magnitude of shear in q points [T-1 ~> s-1]

  real, dimension(SZI_(G),SZJB_(G)), &
      intent(inout) :: smag_x !< biharmonic Smagorinsky model on base level, x-flux
  real, dimension(SZIB_(G),SZJ_(G)), &
      intent(inout) :: smag_y !< biharmonic Smagorinsky model on base level, y-flux

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq
  integer :: i, j
  real :: weight ! Scaling coefficient times interpolation coefficient (0.5)

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB
  weight = scaling_coefficient * 0.5

  ! Smagorinsky model without non-dimensional coefficient
  do J=Jsq-halo,Jeq+halo ; do i=is-halo,ie+halo
    ! lap_vort_x is already multiplied by a mask
    smag_x(i,J) = (weight * lap_vort_x(i,J)) * &
       (CS%grid_sp_q4(I,J) * shear_mag(I,J) + CS%grid_sp_q4(I-1,J) * shear_mag(I-1,J))
  enddo ; enddo

  do j=js-halo,je+halo ; do I=Isq-halo,Ieq+halo
    smag_y(I,j) = (weight * lap_vort_y(I,j)) * &
       (CS%grid_sp_q4(I,J) * shear_mag(I,J) + CS%grid_sp_q4(I,J-1) * shear_mag(I,J-1))
  enddo ; enddo

end subroutine biharmonic_Smagorinsky

!> This function computes the gradients of the vorticity field:
!! d (vort_xy) /dx, d (vort_xy) /dy
!! Then we compute their divergence to get nabla^2 (vort_xy)
!! Then we compute gradient of nabla^2 (vort_xy)
subroutine compute_vorticity_gradients(vort_xy,                          &
                                       vort_x, vort_y,                   &
                                       lap_vort, lap_vort_x, lap_vort_y, &
                                       G, GV, CS, halo)
  type(ocean_grid_type),   intent(in) :: G       !< The ocean's grid structure.
  type(verticalGrid_type), intent(in) :: GV      !< The ocean's vertical grid structure
  type(PG23_CS),           intent(in) :: CS      !< PG23 control structure.
  integer, intent(in) :: halo !< Currently available halo points for vorticity
                              !! in symmetric memory model

  real, dimension(SZIB_(G),SZJB_(G)), &
      intent(in)    :: vort_xy     !< Vertical vorticity (dv/dx - du/dy) [T-1 ~> s-1]
  real, dimension(SZIB_(G),SZJB_(G)), &
      intent(inout) :: lap_vort    !< Laplacian of Vertical vorticity (dv/dx - du/dy) [T-1 L-2 ~> s-1 m-2]
      real, dimension(SZI_(G),SZJB_(G)), &
      intent(inout) :: vort_x      !< x derivative of Vertical vorticity (dv/dx - du/dy) [T-1 L-2 ~> s-1 m-2]
  real, dimension(SZIB_(G),SZJ_(G)), &
      intent(inout) :: vort_y      !< y derivative of Vertical vorticity (dv/dx - du/dy) [T-1 L-2 ~> s-1 m-2]
  real, dimension(SZI_(G),SZJB_(G)), &
      intent(inout) :: lap_vort_x  !< x derivative of Laplacian of Vertical vorticity (dv/dx - du/dy) [T-1 L-2 ~> s-1 m-2]
  real, dimension(SZIB_(G),SZJ_(G)), &
      intent(inout) :: lap_vort_y  !< y derivative of Laplacian of Vertical vorticity (dv/dx - du/dy) [T-1 L-2 ~> s-1 m-2]
    
  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq
  integer :: i, j

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  ! Vorticity x-gradient
  ! This operation does not loose halo on symmetric grid
  do J=Jsq-halo,Jeq+halo ; do i=is-halo,ie+halo
    vort_x(i,J) = G%IdxCv(i,J) * (vort_xy(I,J) - vort_xy(I-1,J)) * G%mask2dCv(i,J)
  enddo ; enddo

  ! Vorticity y-gradient
  do J=js-halo,je+halo ; do I=Isq-halo,Ieq+halo
    vort_y(I,j) = G%IdyCu(I,j) * (vort_xy(I,J) - vort_xy(I,J-1)) * G%mask2dCu(I,j)
  enddo ; enddo

  ! Divergence of vorticity gradients
  ! According to finite volume formula
  do J=Jsq-(halo-1),Jeq+(halo-1) ; do I=Isq-(halo-1),Ieq+(halo-1)
    lap_vort(I,J) = G%IareaBu(I,J) * G%mask2dBu(I,J) * (                         &
                    (vort_x(i+1,J)*G%dyCv(i+1,J) - vort_x(i,J)*G%dyCv(i,J)) +  &
                    (vort_y(I,j+1)*G%dxCu(I,j+1) - vort_y(I,j)*G%dxCu(I,j))    &
                                                                              )
  enddo ; enddo

  ! Lap Vorticity x-gradient
  do J=Jsq-(halo-1),Jeq+(halo-1) ; do i=is-(halo-1),ie+(halo-1)
    lap_vort_x(i,J) = G%IdxCv(i,J) * (lap_vort(I,J) - lap_vort(I-1,J)) * G%mask2dCv(i,J)
  enddo ; enddo

  ! Lap Vorticity y-gradient
  do j=js-(halo-1),je+(halo-1) ; do I=Isq-(halo-1),Ieq+(halo-1)
    lap_vort_y(I,j) = G%IdyCu(I,j) * (lap_vort(I,J) - lap_vort(I,J-1)) * G%mask2dCu(I,j)
  enddo ; enddo

end subroutine compute_vorticity_gradients

!> This function computes the gradients of the velocity field:
!! vort_xy = (dv/dx - du/dy)
!! sh_xy = (du/dy + dv/dx)
!! sh_xx = (du/dx - dv/dy)
!! shear_mag = sqrt(sh_xy**2 + sh_xx**2)
subroutine compute_velocity_gradients(u, v,                    &
                            sh_xx, sh_xy, vort_xy, shear_mag,  &
                            G, GV, CS, halo)
  type(ocean_grid_type),   intent(in) :: G       !< The ocean's grid structure.
  type(verticalGrid_type), intent(in) :: GV      !< The ocean's vertical grid structure
  type(PG23_CS),           intent(in) :: CS      !< PG23 control structure.
  real, dimension(SZIB_(G),SZJ_(G)), &
           intent(in) :: u !< The zonal velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJB_(G)), &
           intent(in) :: v !< The meridional velocity [L T-1 ~> m s-1].
  integer, intent(in) :: halo !< Currently available halo points for u,v velocities
                              !! in symmetric memory model

  real, dimension(SZIB_(G),SZJB_(G)), &
    intent(inout) :: sh_xy       !< horizontal shearing strain (du/dy + dv/dx)
                                 !! including metric terms [T-1 ~> s-1]
  real, dimension(SZIB_(G),SZJB_(G)), &
    intent(inout) :: vort_xy     !< Vertical vorticity (dv/dx - du/dy)
                                 !! including metric terms [T-1 ~> s-1]
  real, dimension(SZI_(G),SZJ_(G)), &
    intent(inout) :: sh_xx       !< horizontal tension (du/dx - dv/dy)
                                 !! including metric terms [T-1 ~> s-1]
  real, dimension(SZIB_(G),SZJB_(G)), &
    intent(inout) :: shear_mag   !< Magnitude of shear in q points [T-1 ~> s-1]

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq
  integer :: i, j

  real :: dudx, dvdy, dvdx, dudy ! Components of velocity gradient tensor [T-1 ~> s-1]
  real :: sh_xx_sq, sh_xy_sq     ! Squares of velocity gradients [T-2 ~> s-2]

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  ! Calculate horizontal tension
  ! Here we assume symmetric memory model, where 
  ! gradients in center points can be computed with same halo as 
  ! velocities
  do j=js-halo,je+halo ; do i=is-halo,ie+halo
    dudx = CS%dy_dxT(i,j)*(G%IdyCu(I,j) * u(I,j) - &
                                G%IdyCu(I-1,j) * u(I-1,j))
    dvdy = CS%dx_dyT(i,j)*(G%IdxCv(i,J) * v(i,J) - &
                                G%IdxCv(i,J-1) * v(i,J-1))
    sh_xx(i,j) = G%mask2dT(i,j) * (dudx - dvdy)
  enddo ; enddo

  ! Components for the shearing strain and vorticity
  ! In symmetric memory model we can compute only current_halo-1 points
  do J=Jsq-(halo-1),Jeq+(halo-1) ; do I=Isq-(halo-1),Ieq+(halo-1)
    dvdx = CS%dy_dxBu(I,J)*(v(i+1,J)*G%IdyCv(i+1,J) - v(i,J)*G%IdyCv(i,J))
    dudy = CS%dx_dyBu(I,J)*(u(I,j+1)*G%IdxCu(I,j+1) - u(I,j)*G%IdxCu(I,j))
    sh_xy(I,J) = G%mask2dBu(I,J) * (dvdx + dudy)

    vort_xy(I,J) = G%mask2dBu(I,J) * G%IareaBu(I,J) * (    &
            (v(i+1,J)*G%dyCv(i+1,J) - v(i,J)*G%dyCv(i,J))  &
          - (u(I,j+1)*G%dxCu(I,j+1) - u(I,j)*G%dxCu(I,j))  &
           )
  enddo ; enddo

  ! Shear magnitude in q point. Halo points correspond to
  ! both halo of sh_xx and sh_xy
  do J=Jsq-(halo-1),Jeq+(halo-1) ; do I=Isq-(halo-1),Ieq+(halo-1)
    sh_xy_sq = sh_xy(I,J)**2
    sh_xx_sq = 0.25 * ( (sh_xx(i,j)**2 + sh_xx(i+1,j+1)**2) &
                      + (sh_xx(i,j+1)**2 + sh_xx(i+1,j)**2) )
    shear_mag(I,J) = sqrt(sh_xx_sq + sh_xy_sq) * G%mask2dBu(I,J)
  enddo ; enddo
  
end subroutine compute_velocity_gradients

!> Wrapper for filter_2D function. The border indices for q and h
!! arrays are substituted. 
!! The input array must have zero B.C. applied. B.C. is applied for output array with
!! multiplying by mask.
subroutine filter_wrapper(G, GV, CS, filter_width, halo, niter, q, h, u, v)
  type(ocean_grid_type),   intent(in) :: G       !< The ocean's grid structure.
  type(verticalGrid_type), intent(in) :: GV      !< The ocean's vertical grid structure
  type(PG23_CS),           intent(in) :: CS      !< PG23 control structure
  real, dimension(SZI_(G),SZJ_(G)),   optional, &
           intent(inout) :: h !< Input/output array in h points [arbitrary]
  real, dimension(SZIB_(G),SZJB_(G)), optional, &
           intent(inout) :: q !< Input/output array in q points [arbitrary]
  real, dimension(SZIB_(G),SZJ_(G)),  optional, &
           intent(inout) :: u !< Input/output array in u points [arbitrary]
  real, dimension(SZI_(G),SZJB_(G)),  optional, &
           intent(inout) :: v !< Input/output array in v points [arbitrary]
  integer, intent(in)    :: halo                 !< Currently available halo points
  integer, intent(in)    :: niter                !< The number of iterations to perform
  real,    intent(in)    :: filter_width         !< Filter width of the filter w.r.t. grid spacing

  if (niter == 0) return ! nothing to do

  call cpu_clock_begin(CS%id_clock_filter)

  if (present(h)) then
    call filter_2D(h, G%mask2dT, filter_width,     &
              G%isd, G%ied, G%jsd, G%jed,          &
              G%isc, G%iec, G%jsc, G%jec,          &
              halo, niter)
  endif

  if (present(q)) then
    call filter_2D(q, G%mask2dBu, filter_width,    &
              G%IsdB, G%IedB, G%JsdB, G%JedB,      &
              G%IscB, G%IecB, G%JscB, G%JecB,      &
              halo, niter)
  endif

  if (present(u)) then
    call filter_2D(u, G%mask2dCu, filter_width,    &
              G%IsdB, G%IedB, G%jsd, G%jed,        &
              G%IscB, G%IecB, G%jsc, G%jec,        &
              halo, niter)
  endif

  if (present(v)) then
    call filter_2D(v, G%mask2dCv, filter_width,    &
              G%isd, G%ied, G%JsdB, G%JedB,        &
              G%isc, G%iec, G%JscB, G%JecB,        &
              halo, niter)
  endif

  call cpu_clock_end(CS%id_clock_filter)
end subroutine filter_wrapper

!> Spatial lateral filter applied to 2D array. The lateral filter is given
!! by the convolutional kernel:
!!     [1 2 1]
!! C = |2 4 2| * 1/16
!!     [1 2 1]
!! The fast algorithm decomposes the 2D filter into two 1D filters as follows:
!!     [1]
!! C = |2| * [1 2 1] * 1/16
!!     [1]
!! The input array must have zero B.C. applied. B.C. is applied for output array with
!! multiplying by mask.
subroutine filter_2D(x, mask, filter_width,                 &
                     isd, ied, jsd, jed, is, ie, js, je,    &
                     current_halo, niter)
  integer, intent(in) :: isd !< Indices of array size
  integer, intent(in) :: ied !< Indices of array size
  integer, intent(in) :: jsd !< Indices of array size
  integer, intent(in) :: jed !< Indices of array size
  integer, intent(in) :: is  !< Indices of owned points
  integer, intent(in) :: ie  !< Indices of owned points
  integer, intent(in) :: js  !< Indices of owned points
  integer, intent(in) :: je  !< Indices of owned points
  real, dimension(isd:ied,jsd:jed), &
           intent(inout) :: x !< Input/output array [arbitrary]
  real, dimension(isd:ied,jsd:jed), &
           intent(in)    :: mask !< Mask array of land points divided by 16 [nondim]
  real,    intent(in)    :: filter_width         !< Filter width of the filter w.r.t. grid spacing
  integer, intent(in)    :: current_halo         !< Currently available halo points
  integer, intent(in)    :: niter                !< The number of iterations to perform

  real :: weight_center, weight_side ! Filter weights [nondim]
  integer :: i, j, k, iter, halo

  real :: tmp(isd:ied, jsd:jed) ! Array with temporary results [arbitrary]

  if (niter == 0) return ! nothing to do
  if (niter > current_halo) then
    call MOM_error(FATAL, &
         "MOM_dynamic_closures: filter_2D requires too many iterations")
  endif

  weight_side = filter_width**2 / 24.
  weight_center = 1. - 2. * weight_side 

  halo = current_halo - 1 ! Save as many halo points as possible
  do iter=1,niter
      do j = js-halo, je+halo; do i = is-halo-1, ie+halo+1
        tmp(i,j) = weight_center * x(i,j) + weight_side * (x(i,j-1) + x(i,j+1))
      enddo; enddo

      do j = js-halo, je+halo; do i = is-halo, ie+halo;
        x(i,j) = (weight_center * tmp(i,j) + weight_side * (tmp(i-1,j) + tmp(i+1,j))) * mask(i,j)
      enddo; enddo
    halo = halo - 1
  enddo

  ! Update halo information
  ! Do this operation in mind whenever you use this function
  ! current_halo = current_halo - niter

end subroutine filter_2D

end module MOM_dynamic_closures