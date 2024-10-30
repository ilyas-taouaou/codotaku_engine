fn main() -> anyhow::Result<()> {
    println!("cargo:rerun-if-changed=devres");
    println!("cargo:rerun-if-changed=res");

    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_3 as u32,
    );
    options.set_source_language(shaderc::SourceLanguage::GLSL);
    options.set_include_callback(|name, _, _, _| {
        let path = format!("devres/shaders/{}", name);
        let source = std::fs::read_to_string(&path).unwrap();
        Ok(shaderc::ResolvedInclude {
            resolved_name: name.to_string(),
            content: source,
        })
    });

    let is_debug_build = std::env::var("OPT_LEVEL")? == "0";

    if is_debug_build {
        options.set_optimization_level(shaderc::OptimizationLevel::Zero);
        options.set_generate_debug_info();
    } else {
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);
    }

    std::fs::create_dir_all("res/shaders")?;

    for entry in std::fs::read_dir("devres/shaders")? {
        let entry = entry?;
        let path = entry.path();
        let extension = path.extension().unwrap().to_str().unwrap();
        let file_name = path.file_name().unwrap().to_str().unwrap();
        let shader_kind = match extension {
            "vert" => shaderc::ShaderKind::Vertex,
            "frag" => shaderc::ShaderKind::Fragment,
            "comp" => shaderc::ShaderKind::Compute,
            "geom" => shaderc::ShaderKind::Geometry,
            "tesc" => shaderc::ShaderKind::TessControl,
            "tese" => shaderc::ShaderKind::TessEvaluation,
            _ => continue,
        };

        let source = std::fs::read_to_string(&path)?;
        let binary_result =
            compiler.compile_into_spirv(&source, shader_kind, file_name, "main", Some(&options))?;

        let binary = binary_result.as_binary_u8();
        let output_path = format!("res/shaders/{}.spv", file_name);
        std::fs::write(output_path, binary)?;
    }

    Ok(())
}
