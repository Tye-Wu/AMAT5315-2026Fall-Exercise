#![no_std]
#![feature(autodiff)]

use core::autodiff::{autodiff_forward, autodiff_reverse};

#[autodiff_forward(cube_forward, Dual, Dual)]
#[autodiff_reverse(cube_reverse, Active, Active)]
fn cube(x: f64) -> f64 { x * x * x }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_cube(x:f64,out:*mut f64) {
    let (y,dy)=cube_forward(x,1.0);
    let (_,adj)=cube_reverse(x,1.0);
    unsafe { *out=y; *out.add(1)=dy; *out.add(2)=adj; }
}

fn step_kernel(nx: usize, nz: usize, dt: f64, dx: f64, c: &[f64], sigma: &[f64],
               prev: &[f64], curr: &[f64], q: &[f64], next: &mut [f64]) {
    let dx2 = dx * dx;
    for i in 0..nx*nz { next[i] = 0.0; }
    for z in 1..nz-1 { for x in 1..nx-1 {
        let i = z * nx + x;
        let lap = (curr[i-1] + curr[i+1] + curr[i-nx] + curr[i+nx] - 4.0 * curr[i]) / dx2;
        let sdt = sigma[i] * dt;
        next[i] = (2.0 * curr[i] - (1.0 - sdt) * prev[i]
            + dt * dt * (c[i] * c[i] * lap + q[i])) / (1.0 + sdt);
    }}
}

#[autodiff_forward(step_jvp, Const, Const, Const, Const, Dual, Const, Dual, Dual, Const, Dual)]
#[autodiff_reverse(step_vjp, Const, Const, Const, Const, Duplicated, Const, Duplicated, Duplicated, Const, Duplicated)]
fn differentiable_step(nx: usize, nz: usize, dt: f64, dx: f64, c: &[f64], sigma: &[f64],
                       prev: &[f64], curr: &[f64], q: &[f64], next: &mut [f64]) {
    step_kernel(nx,nz,dt,dx,c,sigma,prev,curr,q,next)
}

unsafe fn slice<'a>(p: *const f64, n: usize) -> &'a [f64] { unsafe { core::slice::from_raw_parts(p,n) } }
unsafe fn slice_mut<'a>(p: *mut f64, n: usize) -> &'a mut [f64] { unsafe { core::slice::from_raw_parts_mut(p,n) } }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_step_primal(nx: usize, nz: usize, dt: f64, dx: f64,
    c: *const f64, sigma: *const f64, prev: *const f64, curr: *const f64, q: *const f64, next: *mut f64) {
    let n=nx*nz;
    step_kernel(nx,nz,dt,dx,unsafe{slice(c,n)},unsafe{slice(sigma,n)},unsafe{slice(prev,n)},
        unsafe{slice(curr,n)},unsafe{slice(q,n)},unsafe{slice_mut(next,n)});
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_step_jvp(nx: usize, nz: usize, dt: f64, dx: f64,
    c: *const f64, dc: *const f64, sigma: *const f64, prev: *const f64, dprev: *const f64,
    curr: *const f64, dcurr: *const f64, q: *const f64, next: *mut f64, dnext: *mut f64) {
    let n=nx*nz;
    step_jvp(nx,nz,dt,dx,unsafe{slice(c,n)},unsafe{slice(dc,n)},unsafe{slice(sigma,n)},
        unsafe{slice(prev,n)},unsafe{slice(dprev,n)},unsafe{slice(curr,n)},unsafe{slice(dcurr,n)},
        unsafe{slice(q,n)},unsafe{slice_mut(next,n)},unsafe{slice_mut(dnext,n)});
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_step_vjp(nx: usize, nz: usize, dt: f64, dx: f64,
    c: *const f64, dc: *mut f64, sigma: *const f64, prev: *const f64, dprev: *mut f64,
    curr: *const f64, dcurr: *mut f64, q: *const f64, next: *mut f64, dnext: *mut f64) {
    let n=nx*nz;
    step_vjp(nx,nz,dt,dx,unsafe{slice(c,n)},unsafe{slice_mut(dc,n)},unsafe{slice(sigma,n)},
        unsafe{slice(prev,n)},unsafe{slice_mut(dprev,n)},unsafe{slice(curr,n)},unsafe{slice_mut(dcurr,n)},
        unsafe{slice(q,n)},unsafe{slice_mut(next,n)},unsafe{slice_mut(dnext,n)});
}

unsafe extern "C" { fn abort() -> !; }
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { unsafe { abort() } }
