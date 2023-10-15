module Points

using StaticArrays

export Point, PointList, PointMatrix
export sq_norm, dot

const Point = SVector{2, Float64}
const PointList = Vector{Point}
const PointMatrix = Matrix{Point}

function PointList(m::Matrix{T}) where T <: Real 
    size(m, 2) == 2 || error("Dimension mismatch") 
    return copy(
        reinterpret(
            Point, 
            vec(m)
        )
    )
end

sq_norm(x::Point) = x[1]^2 + x[2]^2
dot(x::Point, y::Point) = x[1]*y[1] + x[2]*y[2]

end