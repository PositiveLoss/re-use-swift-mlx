// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "ReuseFastSwift",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(name: "ReuseFastSwift", targets: ["ReuseFastSwift"]),
        .executable(name: "reuse-fast-cli", targets: ["reuse-fast-cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3"),
    ],
    targets: [
        .target(
            name: "ReuseFastSwift",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "reuse-fast-cli",
            dependencies: ["ReuseFastSwift"]
        ),
    ]
)
