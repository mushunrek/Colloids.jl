module Points

using StaticArrays

export Point, PointList, PointMatrix
export sq_norm, dot

const Point = SVector{2, Float64}
const PointList = Vector{Point}
const PointMatrix = Matrix{Point}

sq_norm(x::Point) = x[1]^2 + x[2]^2
dot(x::Point, y::Point) = x[1]*y[1] + x[2]*y[2]

end