//
// Created by eszdman on 29.01.23.
//
#include "Halide.h"
namespace {
using namespace Halide;
using namespace Halide::ConciseCasts;
#define RESX 4000
#define RESY 3000
class GenerateResize : public Halide::Generator<GenerateResize> {
public:
    Input<Buffer<uint16_t, 3>> input{"input"};
    Output<Buffer<uint16_t, 3>> output{"output"};
    Input<int> resizeC{"resizeC"};
    Input<float> smooth{"smoothness"};
    void generate() {
        Var x("x"), y("y"), c("c"), n("n");
        Var tx("tx"), ty("ty"), xi("xi"), yi("yi");
        RDom dim(-1, 3, -1, 3);
        // Add a boundary condition
        Region bounds = {{0,input.width()},{0,input.height()},{0,input.dim(2).extent()}};
        Func clampedInput = BoundaryConditions::mirror_interior(input,bounds);
        Func sumx{"sumx"},sumy{"sumy"},sumK{"sumK"};

        Expr sigma = f32(smooth)*(1.f + f32(resizeC)/2.f);
        Func kernel{"downscale_distribute"};
        kernel(x) = u32_sat(f32(exp(-x*x/(2*sigma*sigma)))*65535.f);
        sumK() = u32(0);
        sumx(x, y, c) = u32(0);
        sumy(x, y, c) = u32(0);
        sumK() += kernel(dim.x);
        Expr norm = sumK();

        Func xReduction("xReduction");
        sumx(x, y, c) += (u32(clampedInput(resizeC * x + dim.x*(1+resizeC/3),y, c)) * kernel(dim.x)/norm);
        xReduction(x, y, c) = sumx(x,y,c);

        sumy(x, y,c) += (xReduction(x, resizeC * y + dim.y, c) * kernel(dim.y)/norm);
        output(x,y,c) = u16_sat(sumy(x,y,c));

        /* ESTIMATES */
        // (This can be useful in conjunction with RunGen and benchmarks as well
        // as auto-schedule, so we do it in all cases.)
        // Provide estimates on the input image

        input.set_estimates({{0, RESX}, {0, RESY},{0,3}});
        // Provide estimates on the output pipeline
        output.set_estimates({{0, RESX/2}, {0, RESY/2},{0,3}});

        resizeC.set_estimate(2);
        smooth.set_estimate(1.f);
        output
        .dim(0).set_stride(3)
        .dim(2).set_stride(1).set_extent(3);
        input
        .dim(0).set_stride(3)
        .dim(2).set_stride(1).set_extent(3);
        /* THE SCHEDULE */
        if (get_auto_schedule()) {
            // nothing
        } else if (get_target().has_gpu_feature()) {
            Var xii, yii;
            std::cout<<"GPU feature"<<std::endl;
            output.compute_root()
                    .reorder(x, y)
                    .gpu_tile(x, y, xi, yi, 8, 8);

        } else {
            std::cout<<"CPU feature"<<std::endl;
            const int vec = natural_vector_size<float>();

            output.compute_root();
        }
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(GenerateResize, resize)