#include <torch/extension.h>       
#include <ATen/cuda/CUDAContext.h> 
#include <ATen/cuda/Atomic.cuh>   
#include <type_traits>

#include "Common.h"
#include <cooperative_groups.h>

namespace mvp_cuda {

namespace cg = cooperative_groups;

template <typename scalar_t>
__device__ void sh_coeffs_to_opacity(
    // inputs
    const uint32_t degree, 
    const vec3 &dir, 
    const scalar_t *coeffs,
    // output
    scalar_t *opacity
) {
    scalar_t pSH0 = 0.2820947917738781f;
    scalar_t sum = pSH0 * coeffs[0];

    if (degree >= 1) {
        scalar_t x = dir.x;
        scalar_t y = dir.y;
        scalar_t z = dir.z;

        scalar_t pSH123 = 0.48860251190292f;
        sum += pSH123 * (-y * coeffs[1] + z * coeffs[2] - x * coeffs[3]);

        if (degree >= 2) {
            scalar_t z2 = z * z;

            scalar_t fTmp0B = -1.092548430592079f * z;
            scalar_t fC1 = x * x - y * y;
            scalar_t fS1 = 2.f * x * y;
            scalar_t pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
            scalar_t pSH7 = fTmp0B * x;
            scalar_t pSH5 = fTmp0B * y;
            scalar_t pSH8 = 0.5462742152960395f * fC1;
            scalar_t pSH4 = 0.5462742152960395f * fS1;

            sum += pSH4 * coeffs[4] + pSH5 * coeffs[5] +
                   pSH6 * coeffs[6] + pSH7 * coeffs[7] +
                   pSH8 * coeffs[8];

            if (degree >= 3) {
                scalar_t fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
                scalar_t fTmp1B = 1.445305721320277f * z;
                scalar_t fC2 = x * fC1 - y * fS1;
                scalar_t fS2 = x * fS1 + y * fC1;
                scalar_t pSH12 =
                    z * (1.865881662950577f * z2 - 1.119528997770346f);
                scalar_t pSH13 = fTmp0C * x;
                scalar_t pSH11 = fTmp0C * y;
                scalar_t pSH14 = fTmp1B * fC1;
                scalar_t pSH10 = fTmp1B * fS1;
                scalar_t pSH15 = -0.5900435899266435f * fC2;
                scalar_t pSH9 = -0.5900435899266435f * fS2;

                sum += pSH9 * coeffs[9] + pSH10 * coeffs[10] +
                       pSH11 * coeffs[11] + pSH12 * coeffs[12] +
                       pSH13 * coeffs[13] + pSH14 * coeffs[14] +
                       pSH15 * coeffs[15];

                if (degree >= 4) {
                    scalar_t fTmp0D =
                        z * (-4.683325804901025f * z2 + 2.007139630671868f);
                    scalar_t fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
                    scalar_t fTmp2B = -1.770130769779931f * z;
                    scalar_t fC3 = x * fC2 - y * fS2;
                    scalar_t fS3 = x * fS2 + y * fC2;
                    scalar_t pSH20 =
                        (1.984313483298443f * z * pSH12 -
                         1.006230589874905f * pSH6);
                    scalar_t pSH21 = fTmp0D * x;
                    scalar_t pSH19 = fTmp0D * y;
                    scalar_t pSH22 = fTmp1C * fC1;
                    scalar_t pSH18 = fTmp1C * fS1;
                    scalar_t pSH23 = fTmp2B * fC2;
                    scalar_t pSH17 = fTmp2B * fS2;
                    scalar_t pSH24 = 0.6258357354491763f * fC3;
                    scalar_t pSH16 = 0.6258357354491763f * fS3;

                    sum += pSH16 * coeffs[16] +
                           pSH17 * coeffs[17] +
                           pSH18 * coeffs[18] +
                           pSH19 * coeffs[19] +
                           pSH20 * coeffs[20] +
                           pSH21 * coeffs[21] +
                           pSH22 * coeffs[22] +
                           pSH23 * coeffs[23] +
                           pSH24 * coeffs[24];
                }
            }
        }
    }

    // sigmoid
    *opacity = 1.0f / (1.0f + expf(-sum));
}

template <typename scalar_t>
__global__ void spherical_harmonics_opacity_fwd_kernel(
    const uint32_t N,
    const uint32_t K,
    const uint32_t degrees_to_use,
    const scalar_t *__restrict__ dirs,       // [N, 3]
    const scalar_t *__restrict__ coeffs, // [N, K, 1]
    scalar_t *__restrict__ opacities        // [N, 1]
) {
    // parallelize over N
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }

    scalar_t dx = dirs[idx * 3 + 0];
    scalar_t dy = dirs[idx * 3 + 1];
    scalar_t dz = dirs[idx * 3 + 2];
    scalar_t dist_sq = dx*dx + dy*dy + dz*dz;
    scalar_t inorm;
    if constexpr (std::is_same_v<scalar_t, float>) {
        inorm = rsqrtf(dist_sq);
    } else {
        inorm = rsqrt(dist_sq);
    }

    vec3 dir = {dx * inorm,
                dy * inorm, 
                dz * inorm};

    sh_coeffs_to_opacity(
        // inputs
        degrees_to_use,
        dir,
        coeffs + idx * K,
        // output
        opacities + idx
    );
}

void launch_spherical_harmonics_opacity_fwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [N, 3]
    const at::Tensor coeffs,              // [N , K, 1]
    // outputs
    at::Tensor opacities // [N, 1]
) {
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = dirs.numel() / 3;

    // parallelize over N
    dim3 threads(256);
    dim3 grid((N + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (N == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        dirs.scalar_type(),
        "spherical_harmonics_opacity_fwd_kernel",
        [&]() {
            spherical_harmonics_opacity_fwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    N,
                    K,
                    degrees_to_use,
                    dirs.data_ptr<scalar_t>(),
                    coeffs.data_ptr<scalar_t>(),
                    opacities.data_ptr<scalar_t>()
                );
        }
    );
}

template <typename scalar_t>
__device__ void sh_coeffs_to_opacity_vjp(
    // fwd inputs
    const uint32_t degree,
    const vec3 &dir, 
    const scalar_t *coeffs, 
    // fwd outputs
    // const float opacity,
    // grad outputs
    const scalar_t v_opacity,
    // grad intputs
    scalar_t *v_coeffs,
    vec3 *v_dir
) {
    scalar_t opacity;
    sh_coeffs_to_opacity(degree, dir, coeffs, &opacity);

    // graident of sigmoid
    scalar_t v_sh_sum = v_opacity * (opacity * (1.0f - opacity));

    v_coeffs[0] += (scalar_t)(v_sh_sum * 0.2820947917738781f);

    if (degree < 1) return;

    scalar_t x = dir.x;
    scalar_t y = dir.y;
    scalar_t z = dir.z;
    scalar_t v_x = 0.f, v_y = 0.f, v_z = 0.f;
    scalar_t pSH123 = v_sh_sum * 0.48860251190292f;

    v_coeffs[1] += (scalar_t)(pSH123 * -y);
    v_coeffs[2] += (scalar_t)(pSH123 * z);
    v_coeffs[3] += (scalar_t)(pSH123 * -x);

    if (v_dir != nullptr) {
        v_x += -pSH123 * coeffs[3];
        v_y += -pSH123 * coeffs[1];
        v_z += pSH123 * coeffs[2];
    }

    if (degree < 2) {
        if (v_dir != nullptr) {
            v_dir->x = v_x;
            v_dir->y = v_y;
            v_dir->z = v_z;
        }
        return;
    }

    scalar_t z2 = z * z;
    scalar_t fTmp0B = -1.092548430592079f * z;
    scalar_t fC1 = x * x - y * y;
    scalar_t fS1 = 2.f * x * y;
    scalar_t pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
    scalar_t pSH7 = fTmp0B * x;
    scalar_t pSH5 = fTmp0B * y;
    scalar_t pSH8 = 0.5462742152960395f * fC1;
    scalar_t pSH4 = 0.5462742152960395f * fS1;
    v_coeffs[4] += (scalar_t)(v_sh_sum * pSH4);
    v_coeffs[5] += (scalar_t)(v_sh_sum * pSH5);
    v_coeffs[6] += (scalar_t)(v_sh_sum * pSH6);
    v_coeffs[7] += (scalar_t)(v_sh_sum * pSH7);
    v_coeffs[8] += (scalar_t)(v_sh_sum * pSH8);

    scalar_t fTmp0B_z, fC1_x, fC1_y, fS1_x, fS1_y, pSH6_z, pSH7_x, pSH7_z, pSH5_y,
        pSH5_z, pSH8_x, pSH8_y, pSH4_x, pSH4_y;
    if (v_dir != nullptr) {
        fTmp0B_z = -1.092548430592079f;
        fC1_x = 2.f * x;
        fC1_y = -2.f * y;
        fS1_x = 2.f * y;
        fS1_y = 2.f * x;
        pSH6_z = 2.f * 0.9461746957575601f * z;
        pSH7_x = fTmp0B;
        pSH7_z = fTmp0B_z * x;
        pSH5_y = fTmp0B;
        pSH5_z = fTmp0B_z * y;
        pSH8_x = 0.5462742152960395f * fC1_x;
        pSH8_y = 0.5462742152960395f * fC1_y;
        pSH4_x = 0.5462742152960395f * fS1_x;
        pSH4_y = 0.5462742152960395f * fS1_y;

        v_x += v_sh_sum *
               (pSH4_x * coeffs[4] + pSH8_x * coeffs[8] +
                pSH7_x * coeffs[7]);
        v_y += v_sh_sum *
               (pSH4_y * coeffs[4] + pSH8_y * coeffs[8] +
                pSH5_y * coeffs[5]);
        v_z += v_sh_sum *
               (pSH6_z * coeffs[6] + pSH7_z * coeffs[7] +
                pSH5_z * coeffs[5]);
    }

    if (degree < 3) {
        if (v_dir != nullptr) {
            v_dir->x = v_x;
            v_dir->y = v_y;
            v_dir->z = v_z;
        }
        return;
    }

    scalar_t fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    scalar_t fTmp1B = 1.445305721320277f * z;
    scalar_t fC2 = x * fC1 - y * fS1;
    scalar_t fS2 = x * fS1 + y * fC1;
    scalar_t pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
    scalar_t pSH13 = fTmp0C * x;
    scalar_t pSH11 = fTmp0C * y;
    scalar_t pSH14 = fTmp1B * fC1;
    scalar_t pSH10 = fTmp1B * fS1;
    scalar_t pSH15 = -0.5900435899266435f * fC2;
    scalar_t pSH9 = -0.5900435899266435f * fS2;
    v_coeffs[9] += (scalar_t)(v_sh_sum * pSH9);
    v_coeffs[10] +=  (scalar_t)(v_sh_sum * pSH10);
    v_coeffs[11] +=  (scalar_t)(v_sh_sum * pSH11);
    v_coeffs[12] +=  (scalar_t)(v_sh_sum * pSH12);
    v_coeffs[13] +=  (scalar_t)(v_sh_sum * pSH13);
    v_coeffs[14] +=  (scalar_t)(v_sh_sum * pSH14);
    v_coeffs[15] +=  (scalar_t)(v_sh_sum * pSH15);


    scalar_t fTmp0C_z, fTmp1B_z, fC2_x, fC2_y, fS2_x, fS2_y, pSH12_z, pSH13_x,
        pSH13_z, pSH11_y, pSH11_z, pSH14_x, pSH14_y, pSH14_z, pSH10_x, pSH10_y,
        pSH10_z, pSH15_x, pSH15_y, pSH9_x, pSH9_y;
    if (v_dir != nullptr) {
        fTmp0C_z = -2.285228997322329f * 2.f * z;
        fTmp1B_z = 1.445305721320277f;
        fC2_x = fC1 + x * fC1_x - y * fS1_x;
        fC2_y = x * fC1_y - fS1 - y * fS1_y;
        fS2_x = fS1 + x * fS1_x + y * fC1_x;
        fS2_y = x * fS1_y + fC1 + y * fC1_y;
        pSH12_z = 3.f * 1.865881662950577f * z2 - 1.119528997770346f;
        pSH13_x = fTmp0C;
        pSH13_z = fTmp0C_z * x;
        pSH11_y = fTmp0C;
        pSH11_z = fTmp0C_z * y;
        pSH14_x = fTmp1B * fC1_x;
        pSH14_y = fTmp1B * fC1_y;
        pSH14_z = fTmp1B_z * fC1;
        pSH10_x = fTmp1B * fS1_x;
        pSH10_y = fTmp1B * fS1_y;
        pSH10_z = fTmp1B_z * fS1;
        pSH15_x = -0.5900435899266435f * fC2_x;
        pSH15_y = -0.5900435899266435f * fC2_y;
        pSH9_x = -0.5900435899266435f * fS2_x;
        pSH9_y = -0.5900435899266435f * fS2_y;

        v_x += v_sh_sum *
               (pSH9_x * coeffs[9] + pSH15_x * coeffs[15] +
                pSH10_x * coeffs[10] + pSH14_x * coeffs[14] +
                pSH13_x * coeffs[13]);

        v_y += v_sh_sum *
               (pSH9_y * coeffs[9] + pSH15_y * coeffs[15] +
                pSH10_y * coeffs[10] + pSH14_y * coeffs[14] +
                pSH11_y * coeffs[11]);

        v_z += v_sh_sum *
               (pSH12_z * coeffs[12] + pSH13_z * coeffs[13] +
                pSH11_z * coeffs[11] + pSH14_z * coeffs[14] +
                pSH10_z * coeffs[10]);
    }

    if (degree < 4) {
        if (v_dir != nullptr) {
            v_dir->x = v_x;
            v_dir->y = v_y;
            v_dir->z = v_z;
        }
        return;
    }

    scalar_t fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    scalar_t fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    scalar_t fTmp2B = -1.770130769779931f * z;
    scalar_t fC3 = x * fC2 - y * fS2;
    scalar_t fS3 = x * fS2 + y * fC2;
    scalar_t pSH20 = (1.984313483298443f * z * pSH12 + -1.006230589874905f * pSH6);
    scalar_t pSH21 = fTmp0D * x;
    scalar_t pSH19 = fTmp0D * y;
    scalar_t pSH22 = fTmp1C * fC1;
    scalar_t pSH18 = fTmp1C * fS1;
    scalar_t pSH23 = fTmp2B * fC2;
    scalar_t pSH17 = fTmp2B * fS2;
    scalar_t pSH24 = 0.6258357354491763f * fC3;
    scalar_t pSH16 = 0.6258357354491763f * fS3;
    v_coeffs[16] += (scalar_t)(v_sh_sum * pSH16);
    v_coeffs[17] += (scalar_t)(v_sh_sum * pSH17);
    v_coeffs[18] += (scalar_t)(v_sh_sum * pSH18);
    v_coeffs[19] += (scalar_t)(v_sh_sum * pSH19);
    v_coeffs[20] += (scalar_t)(v_sh_sum * pSH20);
    v_coeffs[21] += (scalar_t)(v_sh_sum * pSH21);
    v_coeffs[22] += (scalar_t)(v_sh_sum * pSH22);
    v_coeffs[23] += (scalar_t)(v_sh_sum * pSH23);
    v_coeffs[24] += (scalar_t)(v_sh_sum * pSH24);

    scalar_t fTmp0D_z, fTmp1C_z, fTmp2B_z, fC3_x, fC3_y, fS3_x, fS3_y, pSH20_z,
        pSH21_x, pSH21_z, pSH19_y, pSH19_z, pSH22_x, pSH22_y, pSH22_z, pSH18_x,
        pSH18_y, pSH18_z, pSH23_x, pSH23_y, pSH23_z, pSH17_x, pSH17_y, pSH17_z,
        pSH24_x, pSH24_y, pSH16_x, pSH16_y;
    if (v_dir != nullptr) {
        fTmp0D_z = 3.f * -4.683325804901025f * z2 + 2.007139630671868f;
        fTmp1C_z = 2.f * 3.31161143515146f * z;
        fTmp2B_z = -1.770130769779931f;
        fC3_x = fC2 + x * fC2_x - y * fS2_x;
        fC3_y = x * fC2_y - fS2 - y * fS2_y;
        fS3_x = fS2 + y * fC2_x + x * fS2_x;
        fS3_y = x * fS2_y + fC2 + y * fC2_y;
        pSH20_z = 1.984313483298443f * (pSH12 + z * pSH12_z) +
                  -1.006230589874905f * pSH6_z;
        pSH21_x = fTmp0D;
        pSH21_z = fTmp0D_z * x;
        pSH19_y = fTmp0D;
        pSH19_z = fTmp0D_z * y;
        pSH22_x = fTmp1C * fC1_x;
        pSH22_y = fTmp1C * fC1_y;
        pSH22_z = fTmp1C_z * fC1;
        pSH18_x = fTmp1C * fS1_x;
        pSH18_y = fTmp1C * fS1_y;
        pSH18_z = fTmp1C_z * fS1;
        pSH23_x = fTmp2B * fC2_x;
        pSH23_y = fTmp2B * fC2_y;
        pSH23_z = fTmp2B_z * fC2;
        pSH17_x = fTmp2B * fS2_x;
        pSH17_y = fTmp2B * fS2_y;
        pSH17_z = fTmp2B_z * fS2;
        pSH24_x = 0.6258357354491763f * fC3_x;
        pSH24_y = 0.6258357354491763f * fC3_y;
        pSH16_x = 0.6258357354491763f * fS3_x;
        pSH16_y = 0.6258357354491763f * fS3_y;

        v_x += v_sh_sum *
               (pSH16_x * coeffs[16] + pSH24_x * coeffs[24] +
                pSH17_x * coeffs[17] + pSH23_x * coeffs[23] +
                pSH18_x * coeffs[18] + pSH22_x * coeffs[22] +
                pSH21_x * coeffs[21]);
        v_y += v_sh_sum *
               (pSH16_y * coeffs[16] + pSH24_y * coeffs[24] +
                pSH17_y * coeffs[17] + pSH23_y * coeffs[23] +
                pSH18_y * coeffs[18] + pSH22_y * coeffs[22] +
                pSH19_y * coeffs[19]);
        v_z += v_sh_sum *
               (pSH20_z * coeffs[20] + pSH21_z * coeffs[21] +
                pSH19_z * coeffs[19] + pSH22_z * coeffs[22] +
                pSH18_z * coeffs[18] + pSH23_z * coeffs[23] +
                pSH17_z * coeffs[17]);

        if (v_dir != nullptr) {
            v_dir->x = v_x;
            v_dir->y = v_y;
            v_dir->z = v_z;
        }
    }
}

template <typename scalar_t>
__global__ void spherical_harmonics_opacity_bwd_kernel(
    const uint32_t N,
    const uint32_t K,
    const uint32_t degrees_to_use,
    const scalar_t *__restrict__ dirs,         // [N, 3]
    const scalar_t *__restrict__ coeffs,   // [N, K, 1]
    // const scalar_t *__restrict__ opacities, // [N, 1]
    const scalar_t *__restrict__ v_opacities, // [N, 1]
    scalar_t *__restrict__ v_coeffs,       // [N, K, 1]
    scalar_t *__restrict__ v_dirs          // [N, 3] optional
) {
    // parallelize over N
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }

    scalar_t dx = dirs[idx * 3 + 0];
    scalar_t dy = dirs[idx * 3 + 1];
    scalar_t dz = dirs[idx * 3 + 2];
    scalar_t dist_sq = dx*dx + dy*dy + dz*dz;
    scalar_t inorm;
    if constexpr (std::is_same_v<scalar_t, float>) {
        inorm = rsqrtf(dist_sq);
    } else {
        inorm = rsqrt(dist_sq);
    }

    vec3 dir = {dx * inorm,
                dy * inorm, 
                dz * inorm};
    vec3 v_dir = {0.f, 0.f, 0.f};

    sh_coeffs_to_opacity_vjp(
        degrees_to_use,
        dir,
        coeffs + idx * K,
        // opacities[idx],
        v_opacities[idx],
        v_coeffs + idx * K,
        v_dirs == nullptr ? nullptr : &v_dir
    );

    if (v_dirs != nullptr) {
        scalar_t dot = v_dir.x * dir.x + v_dir.y * dir.y + v_dir.z * dir.z;

        scalar_t v_dx = (v_dir.x - dot * dir.x) * inorm;
        scalar_t v_dy = (v_dir.y - dot * dir.y) * inorm;
        scalar_t v_dz = (v_dir.z - dot * dir.z) * inorm;

        v_dirs[idx * 3 + 0] = v_dx;
        v_dirs[idx * 3 + 1] = v_dy;
        v_dirs[idx * 3 + 2] = v_dz;
    }
}

void launch_spherical_harmonics_opacity_bwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [N, 3]
    const at::Tensor coeffs,              // [N, K, 1]
    // const at::Tensor opacities,           // [N, 1]
    const at::Tensor v_opacities,            // [N, 1]
    // outputs
    at::Tensor v_coeffs,            // [N, K, 1]
    at::optional<at::Tensor> v_dirs // [N, 3]
) {
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = dirs.numel() / 3;

    // parallelize over N
    dim3 threads(256);
    dim3 grid((N + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (N == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        dirs.scalar_type(),
        "spherical_harmonics_opacity_bwd_kernel",
        [&]() {
            spherical_harmonics_opacity_bwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    N,
                    K,
                    degrees_to_use,
                    dirs.data_ptr<scalar_t>(),
                    coeffs.data_ptr<scalar_t>(),
                    // opacities.data_ptr<scalar_t>(),
                    v_opacities.data_ptr<scalar_t>(),
                    v_coeffs.data_ptr<scalar_t>(),
                    v_dirs.has_value() ? v_dirs.value().data_ptr<scalar_t>()
                                       : nullptr
                );
        }
    );
}

} // namespace 'mvp_cuda'