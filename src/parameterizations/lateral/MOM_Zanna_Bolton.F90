! > Calculates Zanna and Bolton 2020 parameterization
module MOM_Zanna_Bolton

use MOM_grid,          only : ocean_grid_type
use MOM_verticalGrid,  only : verticalGrid_type
use MOM_diag_mediator, only : diag_ctrl, time_type
use MOM_file_parser,   only : get_param, log_version, param_file_type
use MOM_unit_scaling,  only : unit_scale_type
use MOM_diag_mediator, only : post_data, register_diag_field
use MOM_domains,       only : create_group_pass, do_group_pass, group_pass_type
use MOM_domains,       only : To_North, To_East
use MOM_domains,       only : pass_var, CORNER
use MOM_coms,          only : reproducing_sum

implicit none ; private

#include <MOM_memory.h>

public Zanna_Bolton_2020, ZB_2020_init

!> Control structure that contains MEKE parameters and diagnostics handles
type, public :: ZB2020_CS
  ! Parameters
  logical   :: use_ZB2020    !< If true, parameterization works
  real      :: FGR           !< Filter to grid width ratio, nondimensional, 
                             ! k_bc = - FGR^2 * dx * dy / 24
  integer   :: ZB_type       !< 0 = Zanna Bolton 2020, 1 = Anstey Zanna 2017
  integer   :: ZB_cons       !< 0: nonconservative; 1: conservative without interface;
  integer   :: VGflt_iters   !< number of filter iteration for velocity gradient
  integer   :: VGflt_order   !< order of filtering, 1: Laplacian, 2: Bilaplacian
  integer   :: Strflt_iters  !< number of filter iterations for stress tensor
  integer   :: Strflt_order  !< order of filtering, 1: Laplacian, 2: Bilaplacian
  
  type(diag_ctrl), pointer :: diag => NULL() !< A type that regulates diagnostics output
  !>@{ Diagnostic handles
  integer :: id_ZB2020u = -1, id_ZB2020v = -1, id_KE_ZB2020 = -1, id_kbc = -1
  !>@}

end type ZB2020_CS

contains

subroutine ZB_2020_init(Time, GV, US, param_file, diag, CS)
  type(time_type),         intent(in)    :: Time       !< The current model time.
  type(verticalGrid_type), intent(in)    :: GV   !< The ocean's vertical grid structure
  type(unit_scale_type),   intent(in)    :: US         !< A dimensional unit scaling type
  type(param_file_type),   intent(in)    :: param_file !< Parameter file parser structure.
  type(diag_ctrl), target, intent(inout) :: diag       !< Diagnostics structure.
  type(ZB2020_CS),         intent(inout) :: CS         !< ZB2020 control structure.

  ! This include declares and sets the variable "version".
#include "version_variable.h"
  character(len=40)  :: mdl = "MOM_Zanna_Bolton" ! This module's name.

  call log_version(param_file, mdl, version, "")
  
  call get_param(param_file, mdl, "USE_ZB2020", CS%use_ZB2020, &
                 "If true, turns on Zanna-Bolton 2020 parameterization", &
                 default=.false.)

  call get_param(param_file, mdl, "FGR", CS%FGR, &
                 "The ratio of assumed filter width to grid step", &
                 units="nondim", default=1.)
  
  call get_param(param_file, mdl, "ZB_type", CS%ZB_type, &
                 "Type of parameterization: 0 = ZB2020, 1 = AZ2017", &
                 default=0)

  call get_param(param_file, mdl, "ZB_cons", CS%ZB_cons, &
                 "0: nonconservative; 1: conservative without interface", &
                 default=1)

  call get_param(param_file, mdl, "VGflt_iters", CS%VGflt_iters, &
                 "number of filter iteration for velocity gradient", &
                 default=0)
  
  call get_param(param_file, mdl, "VGflt_order", CS%VGflt_order, &
                 "order of filtering, 1: Laplacian, 2: Bilaplacian", &
                 default=0)

  call get_param(param_file, mdl, "Strflt_iters", CS%Strflt_iters, &
                 "number of filter iterations for stress tensor", &
                 default=0)
  
  call get_param(param_file, mdl, "Strflt_order", CS%Strflt_order, &
                 "order of filtering, 1: Laplacian, 2: Bilaplacian", &
                 default=0)

  ! Register fields for output from this module.
  CS%diag => diag

  CS%id_ZB2020u = register_diag_field('ocean_model', 'ZB2020u', diag%axesCuL, Time, &
      'Zonal Acceleration from Zanna-Bolton 2020', 'm s-2', conversion=US%L_T2_to_m_s2)
  CS%id_ZB2020v = register_diag_field('ocean_model', 'ZB2020v', diag%axesCvL, Time, &
      'Meridional Acceleration from Zanna-Bolton 2020', 'm s-2', conversion=US%L_T2_to_m_s2)
  CS%id_KE_ZB2020 = register_diag_field('ocean_model', 'KE_ZB2020', diag%axesTL, Time, &
      'Kinetic Energy Source from Horizontal Viscosity', &
      'm3 s-3', conversion=GV%H_to_m*(US%L_T_to_m_s**2)*US%s_to_T)
  CS%id_kbc = register_diag_field('ocean_model', 'kbc_ZB2020', diag%axesTL, Time, &
      'Kinetic Energy Source from Horizontal Viscosity', &
      'm2', conversion=US%L_to_m**2)

end subroutine ZB_2020_init

!> Baroclinic parameterization is as follows:
!! eq. 6 in https://laurezanna.github.io/files/Zanna-Bolton-2020.pdf
!! (du/dt, dv/dt) = k_BC * 
!!                  (div(S0) + 1/2 * grad(vort_xy^2 + sh_xy^2 + sh_xx^2))
!! vort_xy = dv/dx - du/dy - relative vorticity
!! sh_xy   = dv/dx + du/dy - shearing deformation (or horizontal shear strain)
!! sh_xx   = du/dx - dv/dy - stretching deformation (or horizontal tension)
!! S0 - 2x2 tensor:
!! S0 = vort_xy * (-sh_xy, sh_xx; sh_xx, sh_xy)
!! Relating k_BC to velocity gradient model,
!! k_BC = - FGR^2 * cell_area / 24 = - FGR^2 * dx*dy / 24
!! where FGR - filter to grid width ratio
!! 
!! S - is a tensor of full tendency
!! S = (-vort_xy * sh_xy + 1/2 * (vort_xy^2 + sh_xy^2 + sh_xx^2), vort_xy * sh_xx;
!!       vort_xy * sh_xx, vort_xy * sh_xy + 1/2 * (vort_xy^2 + sh_xy^2 + sh_xx^2))
!! So the full parameterization:
!! (du/dt, dv/dt) = k_BC * div(S)
!! In generalized curvilinear orthogonal coordinates (see Griffies 2020,
!! and MOM documentation 
!! https://mom6.readthedocs.io/en/dev-gfdl/api/generated/modules/mom_hor_visc.html#f/mom_hor_visc):
!! du/dx -> dy/dx * delta_i (u / dy)
!! dv/dy -> dx/dy * delta_j (v / dx)
!! dv/dx -> dy/dx * delta_i (v / dy)
!! du/dy -> dx/dy * delta_j (u / dx)
!!
!! vort_xy and sh_xy are in the corner of the cell
!! sh_xx in the center of the cell
!!
!! In order to compute divergence of S, its components must be:
!! S_11, S_22 in center of the cells
!! S_12 (=S_21) in the corner
!!
!! The following interpolations are required:
!! sh_xx center -> corner
!! vort_xy, sh_xy corner -> center
subroutine Zanna_Bolton_2020(u, v, h, fx, fy, G, GV, CS)
  type(ocean_grid_type),         intent(in)  :: G      !< The ocean's grid structure.
  type(verticalGrid_type),       intent(in)  :: GV     !< The ocean's vertical grid structure.
  type(ZB2020_CS),               intent(in)  :: CS     !< ZB2020 control structure.

  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
                                 intent(in)  :: u      !< The zonal velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
                                 intent(in)  :: v      !< The meridional velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), &
                                 intent(inout) :: h    !< Layer thicknesses [H ~> m or kg m-2].
  
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
                                 intent(out) :: fx     !< Zonal acceleration due to convergence of
                                                       !! along-coordinate stress tensor [L T-2 ~> m s-2]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
                                 intent(out) :: fy     !< Meridional acceleration due to convergence
                                                       !! of along-coordinate stress tensor [L T-2 ~> m s-2]

  real, dimension(SZI_(G),SZJ_(G)) :: &
    dx_dyT, &          !< Pre-calculated dx/dy at h points [nondim]
    dy_dxT, &          !< Pre-calculated dy/dx at h points [nondim]
    dx2h, &            !< Pre-calculated dx^2 at h points [L2 ~> m2]
    dy2h, &            !< Pre-calculated dy^2 at h points [L2 ~> m2]
    dudx, dvdy, &      ! components in the horizontal tension [T-1 ~> s-1]
    sh_xx, &           ! horizontal tension (du/dx - dv/dy) including metric terms [T-1 ~> s-1]
    vort_xy_center, &  ! vort_xy in the center
    sh_xy_center, &    ! sh_xy in the center
    S_11, S_22         ! flux tensor in the cell center, multiplied with interface height [m^2/s^2 * h]

  real, dimension(SZIB_(G),SZJB_(G)) :: &
    dx_dyBu, &         !< Pre-calculated dx/dy at q points [nondim]
    dy_dxBu, &         !< Pre-calculated dy/dx at q points [nondim]
    dx2q, &            !< Pre-calculated dx^2 at q points [L2 ~> m2]
    dy2q, &            !< Pre-calculated dy^2 at q points [L2 ~> m2]
    dvdx, dudy, &      ! components in the shearing strain [T-1 ~> s-1]
    vort_xy, &         ! Vertical vorticity (dv/dx - du/dy) including metric terms [T-1 ~> s-1]
    sh_xy, &           ! horizontal shearing strain (du/dy + dv/dx) including metric terms [T-1 ~> s-1]
    sh_xx_corner, &    ! sh_xx in the corner
    S_12, &            ! flux tensor in the corner, multiplied with interface height [m^2/s^2 * h]
    hq                 ! harmonic mean of the harmonic means of the u- & v point thicknesses [H ~> m or kg m-2]

  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)) :: &
    kbc_3d             ! k_bc parameter as 3d field [L2 ~> m2]

  real, dimension(SZIB_(G),SZJ_(G)) :: &
    h_u                ! Thickness interpolated to u points [H ~> m or kg m-2].
  real, dimension(SZI_(G),SZJB_(G)) :: &
    h_v                ! Thickness interpolated to v points [H ~> m or kg m-2].

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq, nz
  integer :: i, j, k, n
  
  real :: h_neglect  ! thickness so small it can be lost in roundoff and so neglected [H ~> m or kg m-2]
  real :: h_neglect3 ! h_neglect^3 [H3 ~> m3 or kg3 m-6]
  real :: h2uq, h2vq ! temporary variables [H2 ~> m2 or kg2 m-4].

  real :: sum_sq     ! squared sum, i.e. 1/2*(vort_xy^2 + sh_xy^2 + sh_xx^2)
  real :: vort_sh    ! multiplication of vort_xt and sh_xy

  real :: k_bc ! free constant in parameterization, k_bc < 0, [k_bc] = m^2
  real :: c_h  

  ! Line 407 of MOM_hor_visc.F90
  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  h_neglect  = GV%H_subroundoff ! Line 410 on MOM_hor_visc.F90
  h_neglect3 = h_neglect**3

  fx(:,:,:) = 0.
  fy(:,:,:) = 0.
  kbc_3d(:,:,:) = 0.

  ! Calculate metric terms (line 2119 of MOM_hor_visc.F90)
  do J=js-2,Jeq+1 ; do I=is-2,Ieq+1
    dx2q(I,J) = G%dxBu(I,J)*G%dxBu(I,J) ; dy2q(I,J) = G%dyBu(I,J)*G%dyBu(I,J)
    DX_dyBu(I,J) = G%dxBu(I,J)*G%IdyBu(I,J) ; DY_dxBu(I,J) = G%dyBu(I,J)*G%IdxBu(I,J)
  enddo ; enddo

  ! Calculate metric terms (line 2122 of MOM_hor_visc.F90)
  do j=Jsq-1,Jeq+2 ; do i=Isq-1,Ieq+2
    dx2h(i,j) = G%dxT(i,j)*G%dxT(i,j) ; dy2h(i,j) = G%dyT(i,j)*G%dyT(i,j)
    DX_dyT(i,j) = G%dxT(i,j)*G%IdyT(i,j) ; DY_dxT(i,j) = G%dyT(i,j)*G%IdxT(i,j)
  enddo ; enddo

  do k=1,nz

    sh_xx(:,:) = 0.
    sh_xy(:,:) = 0.
    vort_xy(:,:) = 0.

    ! Calculate horizontal tension (line 590 of MOM_hor_visc.F90)
    do j=Jsq-1,Jeq+2 ; do i=Isq-1,Ieq+2
      dudx(i,j) = DY_dxT(i,j)*(G%IdyCu(I,j) * u(I,j,k) - &
                                  G%IdyCu(I-1,j) * u(I-1,j,k))
      dvdy(i,j) = DX_dyT(i,j)*(G%IdxCv(i,J) * v(i,J,k) - &
                                  G%IdxCv(i,J-1) * v(i,J-1,k))
      sh_xx(i,j) = dudx(i,j) - dvdy(i,j) ! center of the cell
    enddo ; enddo

    ! Components for the shearing strain (line 599 of MOM_hor_visc.F90)
    do J=Jsq-2,Jeq+2 ; do I=Isq-2,Ieq+2
      dvdx(I,J) = DY_dxBu(I,J)*(v(i+1,J,k)*G%IdyCv(i+1,J) - v(i,J,k)*G%IdyCv(i,J))
      dudy(I,J) = DX_dyBu(I,J)*(u(I,j+1,k)*G%IdxCu(I,j+1) - u(I,j,k)*G%IdxCu(I,j))
    enddo ; enddo

    ! Shearing strain with free-slip B.C. (line 751 of MOM_hor_visc.F90)
    ! We use free-slip as cannot guarantee that non-diagonal stress
    ! will accelerate or decelerate currents
    ! Note that as there is no stencil operator, set of indices
    ! is identical to the previous loop, compared to MOM_hor_visc.F90
    do J=Jsq-2,Jeq+2 ; do I=Isq-2,Ieq+2
      sh_xy(I,J) = G%mask2dBu(I,J) * ( dvdx(I,J) + dudy(I,J) ) ! corner of the cell
    enddo ; enddo

    ! Relative vorticity with free-slip B.C. (line 789 of MOM_hor_visc.F90)
    do J=Jsq-2,Jeq+2 ; do I=Isq-2,Ieq+2
      vort_xy(I,J) = G%mask2dBu(I,J) * ( dvdx(I,J) - dudy(I,J) ) ! corner of the cell
    enddo ; enddo

    call low_pass_filter(G, GV, h(:,:,k), CS%VGflt_iters, CS%VGflt_order, q=sh_xy, T=sh_xx)
    call low_pass_filter(G, GV, h(:,:,k), CS%VGflt_iters, CS%VGflt_order, q=vort_xy)

    ! Corner to center interpolation (line 901 of MOM_hor_visc.F90)
    ! lower index as in loop for sh_xy, but minus 1
    ! upper index is identical 
    do j=Jsq-1,Jeq+2 ; do i=Isq-1,Ieq+2
      sh_xy_center(i,j) = 0.25 * ( (sh_xy(I-1,J-1) + sh_xy(I,J)) &
                                 + (sh_xy(I-1,J) + sh_xy(I,J-1)) )
      vort_xy_center(i,j) = 0.25 * ( (vort_xy(I-1,J-1) + vort_xy(I,J)) &
                                   + (vort_xy(I-1,J) + vort_xy(I,J-1)) )
    enddo ; enddo

    ! Center to corner interpolation
    ! lower index as in loop for sh_xx
    ! upper index as in the same loop, but minus 1
    do J=Jsq-1,Jeq+1 ; do I=Isq-1,Ieq+1
      sh_xx_corner(I,J) = 0.25 * ( (sh_xx(i+1,j+1) + sh_xx(i,j)) &
                                 + (sh_xx(i+1,j) + sh_xx(i,j+1)))
    enddo ; enddo

    ! WITH land mask (line 622 of MOM_hor_visc.F90)
    ! Use of mask eliminates dependence on the 
    ! values on land
    do j=js-2,je+2 ; do I=Isq-1,Ieq+1
      h_u(I,j) = 0.5 * (G%mask2dT(i,j)*h(i,j,k) + G%mask2dT(i+1,j)*h(i+1,j,k))
    enddo ; enddo
    do J=Jsq-1,Jeq+1 ; do i=is-2,ie+2
      h_v(i,J) = 0.5 * (G%mask2dT(i,j)*h(i,j,k) + G%mask2dT(i,j+1)*h(i,j+1,k))
    enddo ; enddo

    ! Line 1187 of MOM_hor_visc.F90
    do J=js-1,Jeq ; do I=is-1,Ieq
      h2uq = 4.0 * (h_u(I,j) * h_u(I,j+1))
      h2vq = 4.0 * (h_v(i,J) * h_v(i+1,J))
      hq(I,J) = (2.0 * (h2uq * h2vq)) &
          / (h_neglect3 + (h2uq + h2vq) * ((h_u(I,j) + h_u(I,j+1)) + (h_v(i,J) + h_v(i+1,J))))
    enddo ; enddo

    ! Form S_11 and S_22 tensors
    ! Indices - intersection of loops for
    ! sh_xy_center and sh_xx
    do j=Jsq,Jeq+1 ; do i=Isq,Ieq+1
      if (CS%ZB_type == 0) then
        sum_sq = 0.5 * &
        (vort_xy_center(i,j)**2 + sh_xy_center(i,j)**2 + sh_xx(i,j)**2)
      elseif (CS%ZB_type == 1) then
        sum_sq = 0.
      endif
      
      if (CS%ZB_cons == 1) then
        vort_sh = 0.25 * (                                          &
          (G%areaBu(I-1,J-1) * vort_xy(I-1,J-1) * sh_xy(I-1,J-1)  + &
           G%areaBu(I  ,J  ) * vort_xy(I  ,J  ) * sh_xy(I  ,J  )) + &
          (G%areaBu(I-1,J  ) * vort_xy(I-1,J  ) * sh_xy(I-1,J  )  + &
           G%areaBu(I  ,J-1) * vort_xy(I  ,J-1) * sh_xy(I  ,J-1))   &
          ) * G%IareaT(i,j)
      else if (CS%ZB_cons == 0) then
        vort_sh = vort_xy_center(i,j) * sh_xy_center(i,j)
      endif
      k_bc = - CS%FGR**2 * G%areaT(i,j) / 24.
      S_11(i,j) = k_bc * (- vort_sh + sum_sq)
      S_22(i,j) = k_bc * (+ vort_sh + sum_sq)
    enddo ; enddo

    ! Form S_12 tensor
    ! indices correspond to sh_xx_corner loop
    do J=Jsq-1,Jeq ; do I=Isq-1,Ieq
      vort_sh = vort_xy(I,J) * sh_xx_corner(I,J)
      k_bc = - CS%FGR**2 * G%areaBu(i,j) / 24.
      S_12(I,J) = k_bc * vort_sh
    enddo ; enddo

    call low_pass_filter(G, GV, h(:,:,k), CS%Strflt_iters, CS%Strflt_order, q=S_12, T=S_11)
    call low_pass_filter(G, GV, h(:,:,k), CS%Strflt_iters, CS%Strflt_order, T=S_22)
    !call low_pass_filter(G, GV, h(:,:,k), CS%Strflt_iters, CS%Strflt_order, q=S_12)
    !call smooth_q(G, GV, S_12, h(:,:,k))
    !call smooth_GME(G, GME_flux_q=S_12)
    !call smooth_GME_new(G, GME_flux_h=S_11)
    !call smooth_GME_new(G, GME_flux_h=S_22)
    !call smooth_GME_new(G, GME_flux_q=S_12)

    do j=Jsq,Jeq+1 ; do i=Isq,Ieq+1
      kbc_3d(i,j,k) = - CS%FGR**2 * G%areaT(i,j) / 24.
    enddo ; enddo

    ! Weight with interface height (Line 1478 of MOM_hor_visc.F90)
    ! Note that reduction is removed
    do J=Jsq,Jeq+1 ; do i=Isq,Ieq+1
      S_11(i,j) = S_11(i,j) * h(i,j,k)
      S_22(i,j) = S_22(i,j) * h(i,j,k)
    enddo ; enddo

    ! Free slip (Line 1487 of MOM_hor_visc.F90)
    do J=js-1,Jeq ; do I=is-1,Ieq
      S_12(I,J) = S_12(I,J) * (hq(I,J) * G%mask2dBu(I,J))
    enddo ; enddo

    ! Evaluate 1/h x.Div(h S) (Line 1495 of MOM_hor_visc.F90)
    ! Minus occurs because in original file (du/dt) = - div(S),
    ! but here is the discretization of div(S)
    do j=js,je ; do I=Isq,Ieq
      fx(I,j,k) = - ((G%IdyCu(I,j)*(dy2h(i,j)  *S_11(i,j) - &
                                    dy2h(i+1,j)*S_11(i+1,j)) + &
                      G%IdxCu(I,j)*(dx2q(I,J-1)*S_12(I,J-1) - &
                                    dx2q(I,J)  *S_12(I,J))) * &
                      G%IareaCu(I,j)) / (h_u(I,j) + h_neglect)
    enddo ; enddo

    ! Evaluate 1/h y.Div(h S) (Line 1517 of MOM_hor_visc.F90)
    do J=Jsq,Jeq ; do i=is,ie
      fy(i,J,k) = - ((G%IdyCv(i,J)*(dy2q(I-1,J)*S_12(I-1,J) - &
                                    dy2q(I,J)  *S_12(I,J)) + & ! NOTE this plus
                      G%IdxCv(i,J)*(dx2h(i,j)  *S_22(i,j) - &
                                    dx2h(i,j+1)*S_22(i,j+1))) * &
                      G%IareaCv(i,J)) / (h_v(i,J) + h_neglect)
    enddo ; enddo

  enddo ! end of k loop

  if (CS%id_ZB2020u>0)   call post_data(CS%id_ZB2020u, fx, CS%diag)
  if (CS%id_ZB2020v>0)   call post_data(CS%id_ZB2020v, fy, CS%diag)
  if (CS%id_kbc>0)       call post_data(CS%id_kbc, kbc_3d, CS%diag)

  call compute_energy_source(u, v, h, fx, fy, G, GV, CS)

end subroutine Zanna_Bolton_2020

subroutine low_pass_filter(G, GV, h2d, n_lowpass, n_highpass, q, T)
  type(ocean_grid_type),                        intent(in)    :: G          !< Ocean grid
  type(verticalGrid_type),                      intent(in)    :: GV         !< The ocean's vertical grid structure
  real, dimension(SZIB_(G),SZJB_(G)), optional, intent(inout) :: q          !< any field at q points
  real, dimension(SZI_(G),SZJ_(G)),   optional, intent(inout) :: T          !< any field at T points
  real, dimension(SZI_(G),SZJ_(G)),             intent(inout) :: h2d        !< interface height at T points. It defines mask
  integer,                                      intent(in)    :: n_lowpass  !< number of low-pass iterations 
  integer,                                      intent(in)    :: n_highpass !< number of high-pass iterations 

  integer :: i, j
  real, dimension(SZIB_(G),SZJB_(G)) :: q0, qf ! initial and filtered fields
  real, dimension(SZI_(G),SZJ_(G))   :: T0, Tf ! initial and filtered fields
  
  do j = 1,n_lowpass
    ! construct low-pass filter by 
    ! iterating residual through high-pass filter
    q0(:,:) = q(:,:)
    do i = 1,n_highpass
      qf(:,:) = q0(:,:)
      call smooth_q(G, GV, qf, h2d)
      q0(:,:) = q0(:,:) - qf(:,:) ! save the residual field for the next iteration
    enddo
    q(:,:) = q(:,:) - q0(:,:) ! remove residual
  enddo

  if (present(T)) then
    do j = 1,n_lowpass
      ! construct low-pass filter by 
      ! iterating residual through high-pass filter
      T0(:,:) = T(:,:)
      do i = 1,n_highpass
        Tf(:,:) = T0(:,:)
        call smooth_T(G, GV, Tf, h2d)
        T0(:,:) = T0(:,:) - Tf(:,:) ! save the residual field for the next iteration
      enddo
      T(:,:) = T(:,:) - T0(:,:) ! remove residual
    enddo
  endif
end subroutine low_pass_filter

!> Copy paste from smooth_GME, MOM_hor_visc.F90
!> Function implements zero dirichlet B.C.
subroutine smooth_q(G, GV, q, h2d)
  type(ocean_grid_type),              intent(in)    :: G   !< Ocean grid
  type(verticalGrid_type),            intent(in)    :: GV  !< The ocean's vertical grid structure
  real, dimension(SZIB_(G),SZJB_(G)), intent(inout) :: q   !< any field at q points
  real, dimension(SZI_(G),SZJ_(G)),   intent(inout) :: h2d !< interface height at T points. It defines mask
  ! local variables
  real, dimension(SZIB_(G),SZJB_(G)) :: q_original
  real, dimension(SZIB_(G),SZJB_(G)) :: mask_q ! mask of wet points
  real :: wc, ww, we, wn, ws ! averaging weights for smoothing
  real :: hh ! interface height value to compare
  integer :: i, j, k
  
  hh = GV%Angstrom_m * 2.
  ! compute weights
  ww = 0.125
  we = 0.125
  ws = 0.125
  wn = 0.125
  wc = 1.0 - (ww+we+wn+ws)

  mask_q(:,:) = 0.
  do J = G%JscB-1, G%JecB+1
    do I = G%IscB-1, G%IecB+1
      if (h2d(i+1,j+1) < hh .or. &
          h2d(i  ,j  ) < hh .or. &
          h2d(i+1,j  ) < hh .or. &
          h2d(i  ,j+1) < hh      &
        ) then
          mask_q(I,J) = 0.
        else
          mask_q(I,J) = 1.
        endif
        mask_q(I,J) = mask_q(I,J) * G%mask2dBu(I,J)
    enddo
  enddo

  call pass_var(q, G%Domain, position=CORNER, complete=.true.)
  q_original(:,:) = q(:,:) * mask_q(:,:)
  q(:,:) = 0.
  do J = G%JscB, G%JecB
    do I = G%IscB, G%IecB
      q(I,J) =  wc * q_original(I,J)   &
              + ww * q_original(I-1,J) &
              + we * q_original(I+1,J) &
              + ws * q_original(I,J-1) &
              + wn * q_original(I,J+1)
      q(I,J) = q(I,J) * mask_q(I,J)
    enddo
  enddo
  call pass_var(q, G%Domain, position=CORNER, complete=.true.)
end subroutine smooth_q

subroutine smooth_GME(G, GME_flux_h, GME_flux_q)
  type(ocean_grid_type),                        intent(in)    :: G         !< Ocean grid
  real, dimension(SZI_(G),SZJ_(G)),   optional, intent(inout) :: GME_flux_h!< GME diffusive flux
                                                              !! at h points
  real, dimension(SZIB_(G),SZJB_(G)), optional, intent(inout) :: GME_flux_q!< GME diffusive flux
                                                              !! at q points
  ! local variables
  real, dimension(SZI_(G),SZJ_(G)) :: GME_flux_h_original
  real, dimension(SZIB_(G),SZJB_(G)) :: GME_flux_q_original
  real :: wc, ww, we, wn, ws ! averaging weights for smoothing
  integer :: i, j, k, s
  do s=1,1
    ! Update halos
    if (present(GME_flux_h)) then
      !### Work on a wider halo to eliminate this blocking send!
      call pass_var(GME_flux_h, G%Domain)
      GME_flux_h_original(:,:) = GME_flux_h(:,:)
      ! apply smoothing on GME
      do j = G%jsc, G%jec
        do i = G%isc, G%iec
          ! skip land points
          if (G%mask2dT(i,j)==0.) cycle
          ! compute weights
          ww = 0.125 * G%mask2dT(i-1,j)
          we = 0.125 * G%mask2dT(i+1,j)
          ws = 0.125 * G%mask2dT(i,j-1)
          wn = 0.125 * G%mask2dT(i,j+1)
          wc = 1.0 - (ww+we+wn+ws)
          !### Add parentheses to make this rotationally invariant.
          GME_flux_h(i,j) =  wc * GME_flux_h_original(i,j)   &
                           + ww * GME_flux_h_original(i-1,j) &
                           + we * GME_flux_h_original(i+1,j) &
                           + ws * GME_flux_h_original(i,j-1) &
                           + wn * GME_flux_h_original(i,j+1)
        enddo
      enddo
    endif
    ! Update halos
    if (present(GME_flux_q)) then
      !### Work on a wider halo to eliminate this blocking send!
      call pass_var(GME_flux_q, G%Domain, position=CORNER, complete=.true.)
      GME_flux_q_original(:,:) = GME_flux_q(:,:)
      ! apply smoothing on GME
      do J = G%JscB, G%JecB
        do I = G%IscB, G%IecB
          ! skip land points
          if (G%mask2dBu(I,J)==0.) cycle
          ! compute weights
          ww = 0.125 * G%mask2dBu(I-1,J)
          we = 0.125 * G%mask2dBu(I+1,J)
          ws = 0.125 * G%mask2dBu(I,J-1)
          wn = 0.125 * G%mask2dBu(I,J+1)
          wc = 1.0 - (ww+we+wn+ws)
          !### Add parentheses to make this rotationally invariant.
          GME_flux_q(I,J) =  wc * GME_flux_q_original(I,J)   &
                           + ww * GME_flux_q_original(I-1,J) &
                           + we * GME_flux_q_original(I+1,J) &
                           + ws * GME_flux_q_original(I,J-1) &
                           + wn * GME_flux_q_original(I,J+1)
        enddo
      enddo
      call pass_var(GME_flux_q, G%Domain, position=CORNER, complete=.true.)
    endif
  enddo ! s-loop
end subroutine smooth_GME

!> Apply a 1-1-4-1-1 Laplacian filter one time on GME diffusive flux to reduce any
!! horizontal two-grid-point noise
subroutine smooth_GME_new(G, GME_flux_h, GME_flux_q)
  type(ocean_grid_type),                        intent(in)    :: G         !< Ocean grid
  real, dimension(SZI_(G),SZJ_(G)),   optional, intent(inout) :: GME_flux_h!< GME diffusive flux
                                                              !! at h points
  real, dimension(SZIB_(G),SZJB_(G)), optional, intent(inout) :: GME_flux_q!< GME diffusive flux
                                                              !! at q points
  ! local variables
  real, dimension(SZI_(G),SZJ_(G)) :: GME_flux_h_original
  real, dimension(SZIB_(G),SZJB_(G)) :: GME_flux_q_original
  real :: wc, ww, we, wn, ws ! averaging weights for smoothing
  integer :: i, j, k, s, halosz
  integer :: xh, xq  ! The number of valid extra halo points for h and q points.
  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq
  integer :: num_smooth_gme = 2

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB
  xh = 0 ; xq = 0

  do s=1,num_smooth_gme
    if (present(GME_flux_h)) then
      if (xh < 0) then
        ! Update halos if needed, but avoid doing so more often than is needed.
        halosz = min(G%isc-G%isd, G%jsc-G%jsd, 2+num_smooth_gme-s)
        call pass_var(GME_flux_h, G%Domain, halo=halosz)
        xh = halosz - 2
      endif
      GME_flux_h_original(:,:) = GME_flux_h(:,:)
      ! apply smoothing on GME
      do j=Jsq-xh,Jeq+1+xh ; do i=Isq-xh,Ieq+1+xh
        ! skip land points
        if (G%mask2dT(i,j)==0.) cycle
        ! compute weights
        ww = 0.125 * G%mask2dT(i-1,j)
        we = 0.125 * G%mask2dT(i+1,j)
        ws = 0.125 * G%mask2dT(i,j-1)
        wn = 0.125 * G%mask2dT(i,j+1)
        wc = 1.0 - ((ww+we)+(wn+ws))
        GME_flux_h(i,j) =  wc * GME_flux_h_original(i,j)   &
                         + ((ww * GME_flux_h_original(i-1,j) + we * GME_flux_h_original(i+1,j)) &
                          + (ws * GME_flux_h_original(i,j-1) + wn * GME_flux_h_original(i,j+1)))
      enddo ; enddo
      xh = xh - 1
    endif
    if (present(GME_flux_q)) then
      if (xq < 0) then
        ! Update halos if needed, but avoid doing so more often than is needed.
        halosz = min(G%isc-G%isd, G%jsc-G%jsd, 2+num_smooth_gme-s)
        call pass_var(GME_flux_q, G%Domain, position=CORNER, complete=.true., halo=halosz)
        xq = halosz - 2
      endif
      GME_flux_q_original(:,:) = GME_flux_q(:,:)
      ! apply smoothing on GME
      do J=js-1-xq,je+xq ; do I=is-1-xq,ie+xq
        ! skip land points
        if (G%mask2dBu(I,J)==0.) cycle
        ! compute weights
        ww = 0.125 * G%mask2dBu(I-1,J)
        we = 0.125 * G%mask2dBu(I+1,J)
        ws = 0.125 * G%mask2dBu(I,J-1)
        wn = 0.125 * G%mask2dBu(I,J+1)
        wc = 1.0 - ((ww+we)+(wn+ws))
        GME_flux_q(I,J) =  wc * GME_flux_q_original(I,J)   &
                         + ((ww * GME_flux_q_original(I-1,J) + we * GME_flux_q_original(I+1,J)) &
                          + (ws * GME_flux_q_original(I,J-1) + wn * GME_flux_q_original(I,J+1)))
      enddo ; enddo
      xq = xq - 1
    endif
  enddo ! s-loop
end subroutine smooth_GME_new

!> Function implements zero dirichlet B.C.
!> Smooth of the field defined in the center of the cell
subroutine smooth_T(G, GV, T, h2d)
  type(ocean_grid_type),              intent(in)    :: G   !< Ocean grid
  type(verticalGrid_type),            intent(in)    :: GV  !< The ocean's vertical grid structure
  real, dimension(SZI_(G),SZJ_(G)),   intent(inout) :: T   !< any field at T points
  real, dimension(SZI_(G),SZJ_(G)),   intent(inout) :: h2d !< interface height at T points. It defines mask
  ! local variables
  real, dimension(SZI_(G),SZJ_(G)) :: T_original
  real, dimension(SZI_(G),SZJ_(G)) :: mask_T ! mask of wet points
  real :: wc, ww, we, wn, ws ! averaging weights for smoothing
  real :: hh ! interface height value to compare
  integer :: i, j, k
  
  hh = GV%Angstrom_m * 2.
  ! compute weights
  ww = 0.125
  we = 0.125
  ws = 0.125
  wn = 0.125
  wc = 1.0 - (ww+we+wn+ws)

  mask_T(:,:) = 0.
  do j = G%jsc-1, G%jec+1
    do i = G%isc-1, G%iec+1
      if (h2d(i,j) < hh) then
        mask_T(i,j) = 0.
      else
        mask_T(i,j) = 1.
      endif
        mask_T(i,j) = mask_T(i,j) * G%mask2dT(i,j)
    enddo
  enddo

  call pass_var(T, G%Domain)
  T_original(:,:) = T(:,:) * mask_T(:,:)
  T(:,:) = 0.
  do j = G%jsc, G%jec
    do i = G%isc, G%iec
      T(i,j) =  wc * T_original(i,j)   &
              + ww * T_original(i-1,j) &
              + we * T_original(i+1,j) &
              + ws * T_original(i,j-1) &
              + wn * T_original(i,j+1)
      T(i,j) = T(i,j) * mask_T(i,j)
    enddo
  enddo
  call pass_var(T, G%Domain)
end subroutine smooth_T

! This is copy-paste from MOM_diagnostics.F90, specifically 1125 line
subroutine compute_energy_source(u, v, h, fx, fy, G, GV, CS)
  type(ocean_grid_type),         intent(in)  :: G      !< The ocean's grid structure.
  type(verticalGrid_type),       intent(in)  :: GV     !< The ocean's vertical grid structure.
  type(ZB2020_CS),               intent(in)  :: CS     !< ZB2020 control structure.

  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
                                 intent(in)  :: u      !< The zonal velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
                                 intent(in)  :: v      !< The meridional velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), &
                                 intent(inout) :: h    !< Layer thicknesses [H ~> m or kg m-2].
  
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
                                 intent(in) :: fx     !< Zonal acceleration due to convergence of
                                                      !! along-coordinate stress tensor [L T-2 ~> m s-2]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
                                 intent(in) :: fy     !< Meridional acceleration due to convergence
                                                      !! of along-coordinate stress tensor [L T-2 ~> m s-2]
  
  real :: KE_term(SZI_(G),SZJ_(G),SZK_(GV)) ! A term in the kinetic energy budget
                                 ! [H L2 T-3 ~> m3 s-3 or W m-2]
  real :: tmp(SZI_(G),SZJ_(G),SZK_(GV)) ! temporary array for integration
  real :: KE_u(SZIB_(G),SZJ_(G)) ! The area integral of a KE term in a layer at u-points
                                 ! [H L4 T-3 ~> m5 s-3 or kg m2 s-3]
  real :: KE_v(SZI_(G),SZJB_(G)) ! The area integral of a KE term in a layer at v-points
                                 ! [H L4 T-3 ~> m5 s-3 or kg m2 s-3]

  type(group_pass_type) :: pass_KE_uv !< A handle used for group halo passes

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq, nz
  integer :: i, j, k

  real :: uh !< Transport through zonal faces = u*h*dy,
             !! [H L2 T-1 ~> m3 s-1 or kg s-1].
  real :: vh !< Transport through meridional faces = v*h*dx,
             !! [H L2 T-1 ~> m3 s-1 or kg s-1].
  real :: global_integral !< Global integral of the energy effect of ZB2020 [W]

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB
  
  if (CS%id_KE_ZB2020 > 0) then
    call create_group_pass(pass_KE_uv, KE_u, KE_v, G%Domain, To_North+To_East)
    
    KE_term(:,:,:) = 0.
    tmp(:,:,:) = 0.
    ! Calculate the KE source from Zanna-Bolton2020 [H L2 T-3 ~> m3 s-3].
    do k=1,nz
      KE_u(:,:) = 0.
      KE_v(:,:) = 0.
      do j=js,je ; do I=Isq,Ieq
        uh = u(I,j,k) * 0.5 * (G%mask2dT(i,j)*h(i,j,k) + G%mask2dT(i+1,j)*h(i+1,j,k)) * &
          G%dyCu(I,j)
        KE_u(I,j) = uh * G%dxCu(I,j) * fx(I,j,k)
      enddo ; enddo
      do J=Jsq,Jeq ; do i=is,ie
        vh = v(i,J,k) * 0.5 * (G%mask2dT(i,j)*h(i,j,k) + G%mask2dT(i,j+1)*h(i,j+1,k)) * &
          G%dxCv(i,J)
        KE_v(i,J) = vh * G%dyCv(i,J) * fy(i,J,k)
      enddo ; enddo
      call do_group_pass(pass_KE_uv, G%domain)
      do j=js,je ; do i=is,ie
        KE_term(i,j,k) = 0.5 * G%IareaT(i,j) &
            * (KE_u(I,j) + KE_u(I-1,j) + KE_v(i,J) + KE_v(i,J-1))
        ! copy-paste from MOM_spatial_means.F90, line 42
        tmp(i,j,k) = KE_term(i,j,k) * G%areaT(i,j) * G%mask2dT(i,j)
      enddo ; enddo
    enddo

    global_integral = reproducing_sum(tmp)

    !write(*,*) 'Global energy rate of change [W] for ZB2020:', global_integral

    call post_data(CS%id_KE_ZB2020, KE_term, CS%diag)
  endif

end subroutine compute_energy_source

end module MOM_Zanna_Bolton