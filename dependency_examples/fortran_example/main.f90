! gfortran main.f90 -o main

program main
    real :: x
    real :: y
    integer :: s

    print *, 'main'
    x=2
    y=2.5
    s=2

    x=x * y**s

    print *,"x",x

    print *,"test"
end program main