#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// Function to compute the sink term
double compute_sink(double x, double y, double z)
{
    return 2 * ((1 + x + z) * sin(x + y) - cos(x + y));
}

// Function to compute the source term
double compute_source(double x, double y, double z)
{
    return (1 + x + z) * sin(x + y);
}

// Structure to hold grid information
typedef struct
{
    int global_rank;        // Rank of the process in the global communicator
    int total_processes;    // Total number of processes
    MPI_Comm grid_comm;     // Communicator for the grid
    int grid_dimensions[3]; // Dimensions of the grid
    int row_index;          // Row index of the current process in the grid
    int col_index;          // Column index of the current process in the grid
    int depth_index;        // Depth index of the current process in the grid
    int grid_rank;          // Rank of the process in the grid communicator

    int left_row, right_row;     // Neighbors in the row direction
    int left_col, right_col;     // Neighbors in the column direction
    int left_depth, right_depth; // Neighbors in the depth direction
} GridInfo;

// Structure to hold mesh data
typedef struct
{
    int total_x, total_y, total_z; // Total number of points in each direction
    int local_x, local_y, local_z; // Number of points in each direction for this process

    double step_size;             // Step size in the grid
    double convergence_threshold; // Threshold for convergence

    double global_start_x, global_start_y, global_start_z; // Starting coordinates of the global grid
    double local_start_x, local_start_y, local_start_z;    // Starting coordinates of the local grid

    double *mesh_values;                             // Array to store the mesh values
    double *sink_values;                             // Array to store the sink values
    double (*sink_func)(double, double, double);     // Function to compute the sink term
    double (*boundary_func)(double, double, double); // Function to compute the boundary values
} MeshData;

// Function to initialize the grid
void initialize_grid(GridInfo *grid)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &grid->global_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &grid->total_processes);

    if (grid->global_rank == 0)
    {
        printf("Using %d processes.\n", grid->total_processes);
    }

    // Create a 3D grid of processes
    int dimensions = 3;
    MPI_Dims_create(grid->total_processes, dimensions, grid->grid_dimensions);

    int periodic[3] = {0, 0, 0}; // No periodic boundary conditions
    int reorder = 1;             // Allow reordering of ranks

    MPI_Cart_create(MPI_COMM_WORLD, dimensions, grid->grid_dimensions, periodic, reorder, &grid->grid_comm);

    MPI_Comm_rank(grid->grid_comm, &grid->grid_rank);

    // Get the coordinates of the current process in the grid
    int coords[3];
    MPI_Cart_coords(grid->grid_comm, grid->grid_rank, dimensions, coords);
    grid->row_index = coords[0];
    grid->col_index = coords[1];
    grid->depth_index = coords[2];

    // Find the neighbors in the row direction
    if (grid->row_index != 0)
    {
        coords[0] = grid->row_index - 1;
        MPI_Cart_rank(grid->grid_comm, coords, &grid->left_row);
    }
    else
    {
        grid->left_row = -1; // No neighbor on the left
    }

    if (grid->row_index != grid->grid_dimensions[0] - 1)
    {
        coords[0] = grid->row_index + 1;
        MPI_Cart_rank(grid->grid_comm, coords, &grid->right_row);
    }
    else
    {
        grid->right_row = -1; // No neighbor on the right
    }

    // Find the neighbors in the column direction
    coords[0] = grid->row_index;

    if (grid->col_index != 0)
    {
        coords[1] = grid->col_index - 1;
        MPI_Cart_rank(grid->grid_comm, coords, &grid->left_col);
    }
    else
    {
        grid->left_col = -1; // No neighbor below
    }

    if (grid->col_index != grid->grid_dimensions[1] - 1)
    {
        coords[1] = grid->col_index + 1;
        MPI_Cart_rank(grid->grid_comm, coords, &grid->right_col);
    }
    else
    {
        grid->right_col = -1; // No neighbor above
    }

    // Find the neighbors in the depth direction
    coords[1] = grid->col_index;

    if (grid->depth_index != 0)
    {
        coords[2] = grid->depth_index - 1;
        MPI_Cart_rank(grid->grid_comm, coords, &grid->left_depth);
    }
    else
    {
        grid->left_depth = -1; // No neighbor behind
    }

    if (grid->depth_index != grid->grid_dimensions[2] - 1)
    {
        coords[2] = grid->depth_index + 1;
        MPI_Cart_rank(grid->grid_comm, coords, &grid->right_depth);
    }
    else
    {
        grid->right_depth = -1; // No neighbor in front
    }
}

// Function to set up the mesh data
void setup_mesh_data(MeshData *mesh, GridInfo *grid)
{
    int x = mesh->total_x;
    int y = mesh->total_y;
    int z = mesh->total_z;

    // Ensure each process has at least 8 elements
    if (x * y * z < grid->total_processes * 8 && grid->global_rank == 0)
    {
        fprintf(stderr, "Each process should have at least 8 elements.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Calculate the local sizes for each process
    mesh->local_x = x / grid->grid_dimensions[0] + (grid->row_index < x % grid->grid_dimensions[0] ? 1 : 0);
    mesh->local_y = y / grid->grid_dimensions[1] + (grid->col_index < y % grid->grid_dimensions[1] ? 1 : 0);
    mesh->local_z = z / grid->grid_dimensions[2] + (grid->depth_index < z % grid->grid_dimensions[2] ? 1 : 0);

    // Allocate memory for the mesh and sink values
    mesh->mesh_values = (double *)calloc(mesh->local_x * mesh->local_y * mesh->local_z, sizeof(double));
    mesh->sink_values = (double *)malloc(mesh->local_x * mesh->local_y * mesh->local_z * sizeof(double));

    if (mesh->mesh_values == NULL || mesh->sink_values == NULL)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        MPI_Abort(MPI_COMM_WORLD, 7);
    }

    // Calculate the starting coordinates for the local grid
    double h = mesh->step_size;
    mesh->local_start_x = mesh->global_start_x + grid->row_index * (x / grid->grid_dimensions[0]) * h;
    mesh->local_start_y = mesh->global_start_y + grid->col_index * (y / grid->grid_dimensions[1]) * h;
    mesh->local_start_z = mesh->global_start_z + grid->depth_index * (z / grid->grid_dimensions[2]) * h;
}

// Function to initialize the mesh values
void initialize_mesh_values(MeshData *mesh, GridInfo *grid)
{
    double h = mesh->step_size;

    // Initialize the sink values
    for (int i = 1; i < mesh->local_x - 1; i++)
    {
        for (int j = 1; j < mesh->local_y - 1; j++)
        {
            for (int k = 1; k < mesh->local_z - 1; k++)
            {
                int index = i * mesh->local_y * mesh->local_z + j * mesh->local_z + k;
                mesh->sink_values[index] = h * h * mesh->sink_func(mesh->local_start_x + i * h, mesh->local_start_y + j * h, mesh->local_start_z + k * h);
            }
        }
    }

    // Initialize the boundary values
    if (grid->row_index == 0)
    {
        for (int j = 0; j < mesh->local_y; j++)
        {
            for (int k = 0; k < mesh->local_z; k++)
            {
                int index = 0 * mesh->local_y * mesh->local_z + j * mesh->local_z + k;
                mesh->mesh_values[index] = mesh->boundary_func(mesh->global_start_x, mesh->local_start_y + j * h, mesh->local_start_z + k * h);
            }
        }
    }

    if (grid->row_index == grid->grid_dimensions[0] - 1)
    {
        int i = mesh->local_x - 1;
        for (int j = 0; j < mesh->local_y; j++)
        {
            for (int k = 0; k < mesh->local_z; k++)
            {
                int index = i * mesh->local_y * mesh->local_z + j * mesh->local_z + k;
                mesh->mesh_values[index] = mesh->boundary_func(mesh->local_start_x + i * h, mesh->local_start_y + j * h, mesh->local_start_z + k * h);
            }
        }
    }

    if (grid->col_index == 0)
    {
        for (int i = 0; i < mesh->local_x; i++)
        {
            for (int k = 0; k < mesh->local_z; k++)
            {
                int index = i * mesh->local_y * mesh->local_z + 0 * mesh->local_z + k;
                mesh->mesh_values[index] = mesh->boundary_func(mesh->local_start_x + i * h, mesh->global_start_y, mesh->local_start_z + k * h);
            }
        }
    }

    if (grid->col_index == grid->grid_dimensions[1] - 1)
    {
        int j = mesh->local_y - 1;
        for (int i = 0; i < mesh->local_x; i++)
        {
            for (int k = 0; k < mesh->local_z; k++)
            {
                int index = i * mesh->local_y * mesh->local_z + j * mesh->local_z + k;
                mesh->mesh_values[index] = mesh->boundary_func(mesh->local_start_x + i * h, mesh->local_start_y + j * h, mesh->local_start_z + k * h);
            }
        }
    }

    if (grid->depth_index == 0)
    {
        for (int i = 0; i < mesh->local_x; i++)
        {
            for (int j = 0; j < mesh->local_y; j++)
            {
                int index = i * mesh->local_y * mesh->local_z + j * mesh->local_z + 0;
                mesh->mesh_values[index] = mesh->boundary_func(mesh->local_start_x + i * h, mesh->local_start_y + j * h, mesh->global_start_z);
            }
        }
    }

    if (grid->depth_index == grid->grid_dimensions[2] - 1)
    {
        int k = mesh->local_z - 1;
        for (int i = 0; i < mesh->local_x; i++)
        {
            for (int j = 0; j < mesh->local_y; j++)
            {
                int index = i * mesh->local_y * mesh->local_z + j * mesh->local_z + k;
                mesh->mesh_values[index] = mesh->boundary_func(mesh->local_start_x + i * h, mesh->local_start_y + j * h, mesh->local_start_z + k * h);
            }
        }
    }
}

// Function to perform the Jacobi iteration
void perform_jacobi_iteration(MeshData *mesh, GridInfo *grid)
{
    double *temp_mesh = (double *)malloc(mesh->local_x * mesh->local_y * mesh->local_z * sizeof(double));
    if (temp_mesh == NULL)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        MPI_Abort(MPI_COMM_WORLD, 6);
    }

    double *swap_ptr = mesh->mesh_values;
    mesh->mesh_values = temp_mesh;
    initialize_mesh_values(mesh, grid);
    mesh->mesh_values = swap_ptr;

    if (grid->grid_rank == 0)
    {
        printf("Starting Jacobi\nMax change ith iteration:\n");
    }

    double max_change;
    do
    {
        max_change = 0;

        // Perform the Jacobi iteration
        for (int i = 1; i < mesh->local_x - 1; i++)
        {
            for (int j = 1; j < mesh->local_y - 1; j++)
            {
                for (int k = 1; k < mesh->local_z - 1; k++)
                {
                    int index = i * mesh->local_y * mesh->local_z + j * mesh->local_z + k;
                    int ind_ip = (i + 1) * mesh->local_y * mesh->local_z + j * mesh->local_z + k;
                    int ind_im = (i - 1) * mesh->local_y * mesh->local_z + j * mesh->local_z + k;
                    int ind_jp = i * mesh->local_y * mesh->local_z + (j + 1) * mesh->local_z + k;
                    int ind_jm = i * mesh->local_y * mesh->local_z + (j - 1) * mesh->local_z + k;
                    int ind_kp = i * mesh->local_y * mesh->local_z + j * mesh->local_z + (k + 1);
                    int ind_km = i * mesh->local_y * mesh->local_z + j * mesh->local_z + (k - 1);

                    double change = mesh->mesh_values[index];
                    temp_mesh[index] = (mesh->sink_values[index] + mesh->mesh_values[ind_ip] + mesh->mesh_values[ind_im] + mesh->mesh_values[ind_jp] + mesh->mesh_values[ind_jm] + mesh->mesh_values[ind_kp] + mesh->mesh_values[ind_km]) / 6;
                    change -= temp_mesh[index];
                    if (fabs(change) > max_change)
                    {
                        max_change = fabs(change);
                    }
                }
            }
        }

        // Swap the pointers
        swap_ptr = mesh->mesh_values;
        mesh->mesh_values = temp_mesh;
        temp_mesh = swap_ptr;

        // Communicate with neighbors
        MPI_Request requests[12];
        int request_count = 0;

        if (grid->row_index != 0)
        {
            MPI_Isend(mesh->mesh_values, 1, MPI_DOUBLE, grid->left_row, 0, grid->grid_comm, &requests[request_count++]);
            MPI_Irecv(mesh->mesh_values, 1, MPI_DOUBLE, grid->left_row, 0, grid->grid_comm, &requests[request_count++]);
        }

        if (grid->row_index != grid->grid_dimensions[0] - 1)
        {
            MPI_Isend(mesh->mesh_values, 1, MPI_DOUBLE, grid->right_row, 0, grid->grid_comm, &requests[request_count++]);
            MPI_Irecv(mesh->mesh_values, 1, MPI_DOUBLE, grid->right_row, 0, grid->grid_comm, &requests[request_count++]);
        }

        if (grid->col_index != 0)
        {
            MPI_Isend(mesh->mesh_values, 1, MPI_DOUBLE, grid->left_col, 0, grid->grid_comm, &requests[request_count++]);
            MPI_Irecv(mesh->mesh_values, 1, MPI_DOUBLE, grid->left_col, 0, grid->grid_comm, &requests[request_count++]);
        }

        if (grid->col_index != grid->grid_dimensions[1] - 1)
        {
            MPI_Isend(mesh->mesh_values, 1, MPI_DOUBLE, grid->right_col, 0, grid->grid_comm, &requests[request_count++]);
            MPI_Irecv(mesh->mesh_values, 1, MPI_DOUBLE, grid->right_col, 0, grid->grid_comm, &requests[request_count++]);
        }

        if (grid->depth_index != 0)
        {
            MPI_Isend(mesh->mesh_values, 1, MPI_DOUBLE, grid->left_depth, 0, grid->grid_comm, &requests[request_count++]);
            MPI_Irecv(mesh->mesh_values, 1, MPI_DOUBLE, grid->left_depth, 0, grid->grid_comm, &requests[request_count++]);
        }

        if (grid->depth_index != grid->grid_dimensions[2] - 1)
        {
            MPI_Isend(mesh->mesh_values, 1, MPI_DOUBLE, grid->right_depth, 0, grid->grid_comm, &requests[request_count++]);
            MPI_Irecv(mesh->mesh_values, 1, MPI_DOUBLE, grid->right_depth, 0, grid->grid_comm, &requests[request_count++]);
        }

        // Reduce the maximum change across all processes
        MPI_Allreduce(MPI_IN_PLACE, &max_change, 1, MPI_DOUBLE, MPI_MAX, grid->grid_comm);

        static int iteration_count = 0;
        iteration_count++;

        if (grid->grid_rank == 0 && (iteration_count == 1 || iteration_count % 200 == 0))
        {
            printf("%d: %e\n", iteration_count, max_change);
        }
        else if (grid->grid_rank == 0 && max_change <= mesh->convergence_threshold)
        {
            printf("%d: %e\n", iteration_count, max_change);
        }

        MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

    } while (max_change > mesh->convergence_threshold);

    free(temp_mesh);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    double start_time, end_time;
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    GridInfo grid;
    initialize_grid(&grid);

    MPI_Barrier(MPI_COMM_WORLD);

    MeshData mesh;
    mesh.total_x = 100;                // value for x
    mesh.total_y = 100;                // value for y
    mesh.total_z = 100;                //  value for z
    mesh.step_size = 0.01;             //  value for step size
    mesh.convergence_threshold = 1e-6; // value for convergence threshold
    mesh.global_start_x = 0.0;         // starting x-coordinate
    mesh.global_start_y = 0.0;         // starting y-coordinate
    mesh.global_start_z = 0.0;         // starting z-coordinate
    mesh.sink_func = &compute_sink;
    mesh.boundary_func = &compute_source;

    setup_mesh_data(&mesh, &grid);
    initialize_mesh_values(&mesh, &grid);

    perform_jacobi_iteration(&mesh, &grid);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    if (grid.global_rank == 0)
    {
        printf("Time needed: %f s\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
