#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> // For measuring time

#define NX 100
#define NY 100
#define NZ 100
#define MAX_ITERATIONS 100000
#define TOLERANCE 1e-6

int main()
{
    double h = 0.01;
    double dx = h, dy = h, dz = h;

    // Allocate memory for arrays
    double ***u = (double ***)malloc(NX * sizeof(double **));
    double ***u_new = (double ***)malloc(NX * sizeof(double **));
    double ***f = (double ***)malloc(NX * sizeof(double **));

    for (int i = 0; i < NX; i++)
    {
        u[i] = (double **)malloc(NY * sizeof(double *));
        u_new[i] = (double **)malloc(NY * sizeof(double *));
        f[i] = (double **)malloc(NY * sizeof(double *));
        for (int j = 0; j < NY; j++)
        {
            u[i][j] = (double *)calloc(NZ, sizeof(double));
            u_new[i][j] = (double *)calloc(NZ, sizeof(double));
            f[i][j] = (double *)calloc(NZ, sizeof(double));
        }
    }

    // Initialize the source term and boundary conditions
    for (int i = 0; i < NX; i++)
    {
        for (int j = 0; j < NY; j++)
        {
            for (int k = 0; k < NZ; k++)
            {
                double x = i * dx;
                double y = j * dy;
                double z = k * dz;

                // Source term (right-hand side of Poisson equation)
                f[i][j][k] = 2 * ((1 + x + z) * sin(x + y) - cos(x + y));

                // Boundary conditions (analytical solution at boundaries)
                if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1 || k == 0 || k == NZ - 1)
                {
                    u[i][j][k] = (1 + x + z) * sin(x + y);
                }
            }
        }
    }

    double max_change_history[MAX_ITERATIONS] = {0};
    int iteration;

    clock_t start_time = clock();

    // Perform Jacobi iterations
    for (iteration = 0; iteration < MAX_ITERATIONS; iteration++)
    {
        double max_change = 0;

        for (int i = 1; i < NX - 1; i++)
        {
            for (int j = 1; j < NY - 1; j++)
            {
                for (int k = 1; k < NZ - 1; k++)
                {
                    // Jacobi update formula
                    u_new[i][j][k] = (u[i + 1][j][k] + u[i - 1][j][k] +
                                      u[i][j + 1][k] + u[i][j - 1][k] +
                                      u[i][j][k + 1] + u[i][j][k - 1] -
                                      dx * dx * f[i][j][k]) /
                                     6.0;

                    // Compute the change
                    double change = fabs(u_new[i][j][k] - u[i][j][k]);
                    if (change > max_change)
                    {
                        max_change = change;
                    }
                }
            }
        }

        // Update the solution
        for (int i = 1; i < NX - 1; i++)
        {
            for (int j = 1; j < NY - 1; j++)
            {
                for (int k = 1; k < NZ - 1; k++)
                {
                    u[i][j][k] = u_new[i][j][k];
                }
            }
        }

        // Store the maximum change
        max_change_history[iteration] = max_change;

        // Check for convergence
        if (max_change < TOLERANCE)
        {
            printf("Converged after %d iterations.\n", iteration + 1);
            break;
        }

        // Print progress every 200 iterations
        if (iteration % 200 == 0)
        {
            printf("Iteration %d: Max change = %e\n", iteration + 1, max_change);
        }
    }

    if (iteration == MAX_ITERATIONS)
    {
        printf("Did not converge within the maximum number of iterations.\n");
    }

    printf("Final maximum change: %e\n", max_change_history[iteration]);
    // End measuring time
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Elapsed time: %.2f seconds\n", elapsed_time);

    // Free allocated memory
    for (int i = 0; i < NX; i++)
    {
        for (int j = 0; j < NY; j++)
        {
            free(u[i][j]);
            free(u_new[i][j]);
            free(f[i][j]);
        }
        free(u[i]);
        free(u_new[i]);
        free(f[i]);
    }
    free(u);
    free(u_new);
    free(f);

    return 0;
}
