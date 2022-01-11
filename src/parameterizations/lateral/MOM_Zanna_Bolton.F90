! > Calculates Zanna and Bolton 2020 parameterization
module MOM_Zanna_Bolton

use MOM_grid,                  only : ocean_grid_type
use MOM_verticalGrid,          only : verticalGrid_type

implicit none ; private

#include <MOM_memory.h>

public Zanna_Bolton_2020

contains

!> Baroclinic parameterization is as follows:
!! eq. 6 in https://laurezanna.github.io/files/Zanna-Bolton-2020.pdf
!! (du/dt, dv/dt) = k_BC * 
!!                  (div(S) + 1/2 * grad(vort_xy^2 + sh_xy^2 + sh_xx^2))
!! vort_xy = dv/dx - du/dy - relative vorticity
!! sh_xy   = dv/dx + du/dy - shearing deformation (or horizontal shear strain)
!! sh_xx   = du/dx - dv/dy - stretching deformation (or horizontal tension)
!! S - 2x2 tensor:
!! S = vort_xy * (-sh_xy, sh_xx; sh_xx, sh_xy)
!! In generalized curvilinear orthogonal coordinates (see Griffies 2020,
!! and MOM documentation 
!! https://mom6.readthedocs.io/en/dev-gfdl/api/generated/modules/mom_hor_visc.html#f/mom_hor_visc):
!! du/dx -> dy/dx * delta_i (u / dy)
!! dv/dy -> dx/dy * delta_j (v / dx)
!! dv/dx -> dy/dx * delta_i (v / dy)
!! du/dy -> dx/dy * delta_j (u / dx)
subroutine Zanna_Bolton_2020(u, v, h, fx, fy, G, GV)
  type(ocean_grid_type),         intent(in)  :: G      !< The ocean's grid structure.
  type(verticalGrid_type),       intent(in)  :: GV     !< The ocean's vertical grid structure.
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
                                 intent(in)  :: u      !< The zonal velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
                                 intent(in)  :: v      !< The meridional velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), &
                                 intent(inout) :: h    !< Layer thicknesses [H ~> m or kg m-2].
  
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
                                 intent(out) :: fx  !< Zonal acceleration due to convergence of
                                                       !! along-coordinate stress tensor [L T-2 ~> m s-2]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
                                 intent(out) :: fy  !< Meridional acceleration due to convergence
                                                       !! of along-coordinate stress tensor [L T-2 ~> m s-2].
  
  real, dimension(SZI_(G),SZJ_(G)) :: &
    dx_dyT, &     !< Pre-calculated dx/dy at h points [nondim]
    dy_dxT, &     !< Pre-calculated dy/dx at h points [nondim]
    dudx, dvdy, & ! components in the horizontal tension [T-1 ~> s-1]
    sh_xx         ! horizontal tension (du/dx - dv/dy) including metric terms [T-1 ~> s-1]

  real, dimension(SZIB_(G),SZJB_(G)) :: &
    dx_dyBu, &    !< Pre-calculated dx/dy at q points [nondim]
    dy_dxBu, &    !< Pre-calculated dy/dx at q points [nondim]
    dvdx, dudy, & ! components in the shearing strain [T-1 ~> s-1]
    vort_xy, &    ! Vertical vorticity (dv/dx - du/dy) including metric terms [T-1 ~> s-1]
    sh_xy         ! horizontal shearing strain (du/dy + dv/dx) including metric terms [T-1 ~> s-1]

  integer :: is, ie, js, je, Isq, Ieq, Jsq, Jeq, nz
  integer :: i, j, k, n

  is  = G%isc  ; ie  = G%iec  ; js  = G%jsc  ; je  = G%jec ; nz = GV%ke
  Isq = G%IscB ; Ieq = G%IecB ; Jsq = G%JscB ; Jeq = G%JecB

  ! Calculate dx/dy and dy/dx
  do J=js-2,Jeq+1 ; do I=is-2,Ieq+1
    DX_dyBu(I,J) = G%dxBu(I,J)*G%IdyBu(I,J) ; DY_dxBu(I,J) = G%dyBu(I,J)*G%IdxBu(I,J)
  enddo ; enddo
  do j=Jsq-1,Jeq+2 ; do i=Isq-1,Ieq+2
    DX_dyT(i,j) = G%dxT(i,j)*G%IdyT(i,j) ; DY_dxT(i,j) = G%dyT(i,j)*G%IdxT(i,j)
  enddo ; enddo

  do k=1,nz

    ! Calculate horizontal tension
    do j=Jsq-1,Jeq+2 ; do i=Isq-1,Ieq+2
      dudx(i,j) = DY_dxT(i,j)*(G%IdyCu(I,j) * u(I,j,k) - &
                                  G%IdyCu(I-1,j) * u(I-1,j,k))
      dvdy(i,j) = DX_dyT(i,j)*(G%IdxCv(i,J) * v(i,J,k) - &
                                  G%IdxCv(i,J-1) * v(i,J-1,k))
      sh_xx(i,j) = dudx(i,j) - dvdy(i,j)
    enddo ; enddo

    ! Components for the shearing strain
    do J=Jsq-2,Jeq+2 ; do I=Isq-2,Ieq+2
      dvdx(I,J) = DY_dxBu(I,J)*(v(i+1,J,k)*G%IdyCv(i+1,J) - v(i,J,k)*G%IdyCv(i,J))
      dudy(I,J) = DX_dyBu(I,J)*(u(I,j+1,k)*G%IdxCu(I,j+1) - u(I,j,k)*G%IdxCu(I,j))
    enddo ; enddo

    ! Shearing strain with free-slip B.C. (line 751 of MOM_hor_visc.F90)
    do J=js-2,Jeq+1 ; do I=is-2,Ieq+1
      sh_xy(I,J) = G%mask2dBu(I,J) * ( dvdx(I,J) + dudy(I,J) )
    enddo ; enddo

    ! Relative vorticity with free-slip B.C. (line 789 of MOM_hor_visc.F90)
    do J=Jsq-2,Jeq+2 ; do I=Isq-2,Ieq+2
      vort_xy(I,J) = G%mask2dBu(I,J) * ( dvdx(I,J) - dudy(I,J) )
    enddo ; enddo
    
  enddo ! end of k loop

end subroutine Zanna_Bolton_2020

end module MOM_Zanna_Bolton