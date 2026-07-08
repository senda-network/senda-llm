-keep class network.senda.** { *; }
-keep class uniffi.mesh_ffi.** { *; }
-keepclassmembers class * {
    native <methods>;
}
