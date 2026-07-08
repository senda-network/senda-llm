plugins {
    kotlin("jvm") version "2.0.21"
    application
}

kotlin {
    jvmToolchain(21)
}

group = "network.senda.example"
version = "0.1.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
    implementation("net.java.dev.jna:jna:5.14.0")
}

// Include parent binding sources directly — avoids triggering the Android NDK native build
sourceSets {
    main {
        kotlin {
            srcDir("../../src/main/kotlin/com/senda")
        }
    }
}

application {
    mainClass.set("network.senda.example.ExampleMainKt")
}
