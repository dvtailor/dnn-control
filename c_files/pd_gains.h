#ifndef _PD_GAINS_H
#define _PD_GAINS_H

#define KPZ 2.0  /* proportional term on z         */
#define KDZ 0.7  /* derivative term on z           */
#define KPT 10   /* proportional term on theta     */
#define KPP 0.2  /* proportional term on auxiliary */
#define KDP 0.2  /* derivative term on auxiliary   */

#define SCALING_COEFF 0.5  /* squared neighbourhood radius for using the PD */

/* constants (do not change) */
#define GRAV_ACC 9.81
#define MASS 0.38905

#endif
