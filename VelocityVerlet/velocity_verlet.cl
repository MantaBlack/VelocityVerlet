__kernel void computeForces(__global float4* forces,
							__global float4* currPos,
							__local float4* posCache)
{
	//FLOPS : 2 * numWorkItems * numParticles * 12

	uint gid          = get_global_id(0);
	uint lid          = get_local_id(0);
	uint localSize    = get_local_size(0);
	uint numParticles = get_global_size(0);
	uint otherIdx     = 0;

	//read position and mass for this particle where 4th component is the mass.
	float4 myPos = currPos[gid];
	float4 force = (float4) 0.0f;

	for (uint gidx = lid; gidx < numParticles; gidx += localSize)
	{
		//cache position and mass for other particles.
		posCache[lid] = currPos[gidx];
		barrier(CLK_LOCAL_MEM_FENCE);

		for (uint q = 0; q < localSize; ++q)
		{
			float4 otherPos = posCache[q];

			float4 diff   = otherPos - myPos;
			float sqrDist = diff.s0 * diff.s0 + diff.s1 * diff.s1 + diff.s2 * diff.s2;
			float gravity = myPos.s3 * otherPos.s3 / (sqrt(sqrDist) * sqrDist);
			force         = select(force + (gravity * diff), force, (uint4)gid == otherIdx);

			++otherIdx;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//write forces so that we can use it later to update positions after syncing.
	forces[gid] = force;
}

__kernel void updatePositions(__global float4* forces,
							  __global float4* currPos,
							  __global float4* newPos,
							  __global float4* currVel,
							  float timeStep)
{
	//FLOPS : numWorkItems * 6

	uint gid = get_global_id(0);

	//read position and mass for this particle, 4th component is mass.
	float4 myPos = currPos[gid];

	//make a copy of the mass so we don't lose it during vector operations.
	float myMass = myPos.s3;

	float4 myVel = currVel[gid];
	float4 myForce = forces[gid];

	//compute new position for this particle.
	float acc = timeStep * 0.5f / myMass;
	myPos = myPos + timeStep * (myVel + acc * myForce);

	//replace mass for this particle.
	myPos.s3 = myMass;

	//write new position.
	newPos[gid] = myPos;
}

__kernel void updateVelocities(__global float4* oldForces,
							   __global float4* newForces,
							   __global float4* currVel,
							   float timeStep)
{
	//FLOPS : numWorkItems * 5

	uint gid       = get_global_id(0);
	uint lid       = get_local_id(0);
	uint localSize = get_local_size(0);

	//read velocity for this particle, 4th component is mass.
	float4 myVel = currVel[gid];

	//make copy of mass so we don't lose it during vector operations.
	float myMass = myVel.s3;

	//old force for this particle.
	float4 myOldForce = oldForces[gid];

	//current force for this particle.
	float4 myNewForce = newForces[gid];

	//update velocity.
	float acc = timeStep * 0.5f / myMass;

	myVel = myVel + acc *(myOldForce + myNewForce);

	myVel.s3 = myMass;
	currVel[gid] = myVel;
}



__kernel void computeForcesNC(__global float4* forces,
							  __global float4* currPos)
{
	uint gid          = get_global_id(0);
	uint lid          = get_local_id(0);
	uint localSize    = get_local_size(0);
	uint numParticles = get_global_size(0);

	//read position and mass for this particle where 4th component is the mass.
	float4 myPos = currPos[gid];
	float4 force = (float4) 0.0f;

	//for all particles
	for (uint q = 0; q < numParticles; ++q)
	{
		float4 otherPos = currPos[q];

		float4 diff   = otherPos - myPos;
		float sqrDist = diff.s0 * diff.s0 + diff.s1 * diff.s1 + diff.s2 * diff.s2;
		float gravity = myPos.s3 * otherPos.s3 / (sqrt(sqrDist) * sqrDist);
		force         = select(force + (gravity * diff), force, (uint4) gid == q);
	}

	//write forces so that we can use it later to update positions after syncing.
	forces[gid] = force;
}
