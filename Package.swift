// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "CotabbyInference",
    platforms: [.macOS(.v14)],
    products: [
        .library(
            name: "CotabbyInference",
            targets: ["CotabbyInferenceEngine"]
        ),
    ],
    targets: [
        .binaryTarget(
            name: "llama-cpp",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b9310/llama-b9310-xcframework.zip",
            checksum: "e2411e2e1a875d38d7e1cd478ea5ba2db1b70817bcd36c624f2e952fd017eb83"
        ),
        .target(
            name: "CotabbyInferenceEngine",
            dependencies: ["llama-cpp"],
            path: "Sources/CotabbyInferenceEngine",
            publicHeadersPath: "include",
            cxxSettings: [
                .unsafeFlags(["-std=c++17"]),
            ]
        ),
        .testTarget(
            name: "CotabbyInferenceTests",
            dependencies: ["CotabbyInferenceEngine"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        ),
        // Phase 0 spike: standalone executable that compares aggregate decode
        // tok/s between the current "separate llama_context per sequence"
        // architecture and a prospective shared-context-with-batched-decode
        // architecture. Used to decide whether the Phase 1 refactor is worth
        // doing on M-series + Metal. Not linked into CotabbyInference itself.
        .executableTarget(
            name: "CotabbyInferenceBench",
            dependencies: ["llama-cpp"],
            path: "Sources/CotabbyInferenceBench",
            cxxSettings: [
                .unsafeFlags(["-std=c++17"]),
            ]
        ),
    ]
)
