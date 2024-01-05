__kernel void compute_forces(__global float4* forces,
    __global float4* curr_positions,
    __local float4* positions_cache)
{
    //FLOPS : 2 * numWorkItems * num_particles * 12

    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint local_size = get_local_size(0);
    uint num_particles = get_global_size(0);

    //read position and mass for this particle where 4th component is the mass.
    float4 my_pos = curr_positions[gid];
    float3 force = (float3)0.0f;

    for (uint other = gid + 1; other < num_particles; ++other)
    {
        float4 other_position = curr_positions[other];

        float3 diff   = other_position.s012 - my_pos.s012;
        float gravity = my_pos.s3 * other_position.s3 / (fast_length(diff) * dot(diff, diff));

        force += (gravity * diff);
    }

    // since atomics for floats is not supported, we need to calculate
    // the opposite forces as follows
    for (int other = gid; other >= 0; --other)
    {
        float4 other_position = curr_positions[other];

        float3 diff   = my_pos.s012 - other_position.s012;
        float gravity = my_pos.s3 * other_position.s3 / (fast_length(diff) * dot(diff, diff));

        force = select(force, force - (gravity * diff), (uint3)other < gid);
    }

    //write forces so that we can use it later to update positions after syncing.
    forces[gid] = (float4)(force, 0.f);
}

__kernel void compute_positions(__global float4* forces,
    __global float4* curr_positions,
    __global float4* new_positions,
    __global float4* current_velocities,
    float time_step)
{
    //FLOPS : numWorkItems * 6

    uint gid = get_global_id(0);

    //read position and mass for this particle, 4th component is mass.
    float4 my_pos = curr_positions[gid];

    //make a copy of the mass so we don't lose it during vector operations.
    float my_mass = my_pos.s3;

    float4 my_velocity = current_velocities[gid];
    float4 my_force = forces[gid];

    //compute new position for this particle.
    float acc = time_step * 0.5f / my_mass;
    my_pos.s012 = my_pos.s012 + time_step * (my_velocity.s012 + acc * my_force.s012);

    //replace mass for this particle.
    my_pos.s3 = my_mass;

    //write new position.
    new_positions[gid] = my_pos;
}

__kernel void compute_velocities(__global float4* oldForces,
    __global float4* new_forces,
    __global float4* current_velocities,
    float time_step)
{
    //FLOPS : numWorkItems * 5

    uint gid        = get_global_id(0);
    uint lid        = get_local_id(0);
    uint local_size = get_local_size(0);

    //read velocity for this particle, 4th component is mass.
    float4 my_velocity = current_velocities[gid];

    //make copy of mass so we don't lose it during vector operations.
    float my_mass = my_velocity.s3;

    //old force for this particle.
    float4 myOldForce = oldForces[gid];

    //current force for this particle.
    float4 myNewForce = new_forces[gid];

    //update velocity.
    float acc = time_step * 0.5f / my_mass;

    my_velocity = my_velocity + acc * (myOldForce + myNewForce);

    my_velocity.s3 = my_mass;
    current_velocities[gid] = my_velocity;
}
