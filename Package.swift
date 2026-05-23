// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "TabbyInference",
    platforms: [.macOS(.v14)],
    products: [
        .library(
            name: "TabbyInference",
            targets: ["TabbyInferenceEngine"]
        ),
    ],
    targets: [
        .binaryTarget(
            name: "llama-cpp",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b8665/llama-b8665-xcframework.zip",
            checksum: "5279c975a0ad136eb0ca29bb6390735b949bc0bed0f803124538e341315cb8f7"
        ),
        .target(
            name: "TabbyInferenceEngine",
            dependencies: ["llama-cpp"],
            path: "Sources/TabbyInferenceEngine",
            publicHeadersPath: "include",
            cxxSettings: [
                .unsafeFlags(["-std=c++17"]),
            ]
        ),
        .testTarget(
            name: "TabbyInferenceTests",
            dependencies: ["TabbyInferenceEngine"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        ),
    ]
)
