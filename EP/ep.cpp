/*--------------------------------------------------------------------

        Information on NAS Parallel Benchmarks is available at:

        http://www.nas.nasa.gov/Software/NPB/

        Authors: P. O. Frederickson
           D. H. Bailey
           A. C. Woo

        STL version:
        Nicco Mietzsch <nicco.mietzsch@campus.lmu.de>

        CPP and OpenMP version:
        Dalvan Griebler <dalvangriebler@gmail.com>
        Júnior Löff <loffjh@gmail.com>

--------------------------------------------------------------------*/
#include <libdash.h>

#include <algorithm>
#include <mutex>
#include <numeric>
#include <vector>
#include <alpaka/alpaka.hpp>
#include <mephisto/algorithm/for_each>
#include <mephisto/entity>
#include <mephisto/execution>
#include <patterns/local_pattern.h>

#include "../common/npb-CPP.hpp"
#include <iostream>
#include "npbparams.hpp"


//#include "../common/mystl.h"

/* parameters */
#define MK 16
#define MM (M - MK)
#define NN (1 << MM)
#define NK (1 << MK)
#define NQ 10
#define EPSILON 1.0e-8
#define A 1220703125.0
#define S 271828183.0
#define TIMERS_ENABLED FALSE

struct ep_res {
  double sx;
  double sy;
  double q[NQ];

  ep_res()
    : sx(0)
    , sy(0)
    , q()
  {
  }
};

/*--------------------------------------------------------------------
    program EMBAR
c-------------------------------------------------------------------*/
/*
c   This is the serial version of the APP Benchmark 1,
c   the "embarassingly parallel" benchmark.
c
c   M is the Log_2 of the number of complex pairs of uniform (0, 1) random
c   numbers.  MK is the Log_2 of the size of each batch of uniform random
c   numbers.  MK can be set for convenience on a given system, since it does
c   not affect the results.
*/
int main(int argc, char **argv)
{
  dash::init(&argc, &argv);

  double  Mops, t1, sx, sy, tm, an, gc;
  double  dum[3] = {1.0, 1.0, 1.0};
  int     nit, k_offset, j;
  boolean verified;

  int nprocs = dash::size();

  if (0 == dash::myid()) {
    std::size_t size = pow(2.0, M + 1);
    printf(" Number of random numbers generated: %zu\n", size);
    verified = FALSE;
  }

  const auto np = NN;

  /*
  c   Call the random number generator functions and initialize
  c   the x-array to reduce the effects of paging on the timings.
  c   Also, call all mathematical functions that are used. Make
  c   sure these initializations cannot be eliminated as dead code.
  */
  vranlc(0, &(dum[0]), dum[1], &(dum[2]));
  dum[0] = randlc(&(dum[1]), dum[2]);

	std::vector<double> my_x;
	my_x.resize(2 * NK + 1);

	std::fill(my_x.begin(), my_x.end(), -1.0e99);

  if (0 == dash::myid()) {
    timer_clear(1);
    timer_clear(2);
    timer_clear(3);
    timer_clear(4);
    timer_start(1);
  }

  vranlc(0, &t1, A, my_x.data());

  /*   Compute AN = A ^ (2 * NK) (mod 2^46). */

  t1 = A;

  for (int i = 1; i <= MK + 1; i++) {
    an = randlc(&t1, t1);
  }


  an              = t1;
  gc              = 0.0;

  if (TIMERS_ENABLED == TRUE) timer_start(4);

  using Data     = int;
  auto const Dim = 1;
  using EntityT =
      mephisto::Entity<Dim, std::size_t, alpaka::acc::AccCpuSerial>;
  using Queue   = alpaka::queue::QueueCpuSync;
  using Context = mephisto::execution::AlpakaExecutionContext<EntityT, Queue>;
  using BasePattern = dash::BlockPattern<Dim>;
  using PatternT    = patterns::BalancedLocalPattern<BasePattern, EntityT>;

  using DataArray = dash::Array<Data, dash::default_index_t, PatternT>;
  using ResArray = dash::Array<ep_res, dash::default_index_t, PatternT>;

  BasePattern base{np};
  PatternT pattern{base};

  ResArray  res{pattern};
  DataArray v{pattern};

  dash::generate_with_index(v.begin(), v.end(), [](auto i) { return i; });
  v.barrier();

  // Context consists of the host, the accelerator and the stream
  Context ctx;

  // The executor is the one actually doing the computation
  mephisto::execution::AlpakaExecutor<Context> executor{ctx};

  // The policy is used to relax guarantees.
  auto policy = mephisto::execution::make_parallel_policy(executor);

  dash::transform(policy, v.begin(), v.end(), res.begin(), [an, &my_x](int k) {
    double t1, t2, t3, t4, x1, x2;
    int    kk, i, ik, l;

    kk = k;
    t1 = S;
    t2 = an;

    //  Find starting seed t1 for this kk.

    for (i = 1; i <= 100; i++) {
      ik = kk / 2;
      if (2 * ik != kk) t3 = randlc(&t1, t2);
      if (ik == 0) break;
      t3 = randlc(&t2, t2);
      kk = ik;
    }

    //	Compute uniform pseudorandom numbers.

    // if (TIMERS_ENABLED == TRUE) timer_start(3);
    vranlc(2 * NK, &t1, A, my_x.data());
    // if (TIMERS_ENABLED == TRUE) timer_stop(3);

    //
    // c	   Compute Gaussian deviates by acceptance-rejection method
    // and c	   tally counts in concentric square annuli.  This loop is not
    // c	   vectorizable.
    //
    // if (TIMERS_ENABLED == TRUE) timer_start(2);
    //
    ep_res ret;

    for (i = 1; i <= NK; i++) {
      x1 = 2.0 * my_x[2 * i - 1] - 1.0;
      x2 = 2.0 * my_x[2 * i] - 1.0;
      t1 = pow2(x1) + pow2(x2);
      if (t1 <= 1.0) {
        t2 = sqrt(-2.0 * log(t1) / t1);
        t3 = (x1 * t2);  // Xi
        t4 = (x2 * t2);  // Yi
        l  = max(fabs(t3), fabs(t4));
        ret.q[l] += 1.0;                // counts
        ret.sx += t3;  // sum of Xi
        ret.sy += t4;  // sum of Yi
      }
    }

    return ret;
    // if (TIMERS_ENABLED == TRUE) timer_stop(2);
  });

  v.barrier();

  if (TIMERS_ENABLED == TRUE) timer_stop(4);

  ep_res init;

  res[0] = dash::reduce(res.begin(), res.end(), init, [](ep_res a, ep_res b) {
    ep_res c;
    c.sx = a.sx + b.sx;
    c.sy = a.sy + b.sy;
    for (int i = 0; i < NQ; ++i) c.q[i] = a.q[i] + b.q[i];
    return c;
  });

  res.barrier();

  if (0 == dash::myid()) { /*

         for( auto i = 1; i < dash::size(); ++i) {

                 res.local[0].sx += ((ep_res) res[i]).sx;
                 res.local[0].sy += ((ep_res) res[i]).sy;

                 for(j=0; j < NQ; ++j) {
                         res.local[0].q[j] += ((ep_res) res[i]).q[j];
                 }
         }*/

    for (std::size_t i = 0; i <= NQ - 1; i++) {
      gc = gc + res.local[0].q[i];
    }
    sx = res.local[0].sx;
    sy = res.local[0].sy;

    timer_stop(1);
    tm = timer_read(1);

    nit = 0;
    if (M == 24) {
      if ((fabs((sx - (-3.247834652034740e3)) / -3.247834652034740e3) <=
           EPSILON) &&
          (fabs((sy - (-6.958407078382297e3)) / -6.958407078382297e3) <=
           EPSILON)) {
        verified = TRUE;
      }
    }
    else if (M == 25) {
      if ((fabs((sx - (-2.863319731645753e3)) / -2.863319731645753e3) <=
           EPSILON) &&
          (fabs((sy - (-6.320053679109499e3)) / -6.320053679109499e3) <=
           EPSILON)) {
        verified = TRUE;
      }
    }
    else if (M == 28) {
      // if ((fabs((sx- (-4.295875165629892e3))/sx) <= EPSILON) && (fabs((sy-
      // (-1.580732573678431e4))/sy) <= EPSILON)) {
      if ((fabs((sx - (-4.295875165629892e3)) / -4.295875165629892e3) <=
           EPSILON) &&
          (fabs((sy - (-1.580732573678431e4)) / -1.580732573678431e4) <=
           EPSILON)) {
        verified = TRUE;
      }
    }
    else if (M == 30) {
      if ((fabs((sx - (4.033815542441498e4)) / 4.033815542441498e4) <=
           EPSILON) &&
          (fabs((sy - (-2.660669192809235e4)) / -2.660669192809235e4) <=
           EPSILON)) {
        verified = TRUE;
      }
    }
    else if (M == 32) {
      if ((fabs((sx - (4.764367927995374e4)) / 4.764367927995374e4) <=
           EPSILON) &&
          (fabs((sy - (-8.084072988043731e4)) / -8.084072988043731e4) <=
           EPSILON)) {
        verified = TRUE;
      }
    }
    else if (M == 36) {
      if ((fabs((sx - (1.982481200946593e5)) / 1.982481200946593e5) <=
           EPSILON) &&
          (fabs((sy - (-1.020596636361769e5)) / -1.020596636361769e5) <=
           EPSILON)) {
        verified = TRUE;
      }
    }
    else if (M == 40) {
      if ((fabs((sx - (-5.319717441530e5)) / -5.319717441530e5) <= EPSILON) &&
          (fabs((sy - (-3.688834557731e5)) / -3.688834557731e5) <= EPSILON)) {
        verified = TRUE;
      }
    }

    Mops = pow(2.0, M + 1) / tm / 1000000.0;

    printf(
        "EP Benchmark Results: \n"
        "CPU Time = %10.4f\n"
        "N = 2^%5d\n"
        "No. Gaussian Pairs = %15.0f\n"
        "Sums = %25.15e %25.15e\n"
        "Counts:\n",
        tm,
        M,
        gc,
        sx,
        sy);
    for (std::size_t i = 0; i <= NQ - 1; i++) {
      printf("%zu %15.0f\n", i, res.local[0].q[i]);
    }

    c_print_results(
        (char *)"EP",
        CLASS,
        M + 1,
        0,
        0,
        nit,
        nprocs,
        tm,
        Mops,
        (char *)"Random numbers generated",
        verified,
        (char *)NPBVERSION,
        (char *)COMPILETIME,
        (char *)CS1,
        (char *)CS2,
        (char *)CS3,
        (char *)CS4,
        (char *)CS5,
        (char *)CS6,
        (char *)CS7);

    if (TIMERS_ENABLED == TRUE) {
      printf("Total time:	 %f\n", timer_read(1));
      printf("Gaussian pairs: %f\n", timer_read(2));
      printf("Random numbers: %f\n", timer_read(3));
      printf("Time in STL:	%f\n", timer_read(4));
    }
  }

  dash::finalize();

  return 0;
}
