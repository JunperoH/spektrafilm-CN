from dataclasses import dataclass

from spektrafilm_gui.analysis_panel import AnalysisPanel
from spektrafilm_gui.widget_editors import BoolEditor, EnumEditor, FloatEditor, FloatTupleEditor, IntEditor, IntTupleEditor
from spektrafilm_gui.widget_primitives import CollapsibleSection, platform_default_font
from spektrafilm_gui.paper_section import PaperTextureSection
from spektrafilm_gui.widget_sections import (
    AgingSection,
    CameraSection,
    CameraDiffusionSection,
    ChemistrySection,
    CouplersSection,
    DataclassSection,
    DisplaySection,
    DiffusionSection,
    EnlargerSection,
    ExportSection,
    ExposureControlSection,
    FilePickerSection,
    FilmStripWidget,
    GlareSection,
    InputGamutCompressSection,
    LocationsSection,
    OutputGamutCompressSection,
    GpuBackendWidget,
    BleachBypassSection,
    ConfigDefaultsSection,
    DevelopmentSection,
    GrainSection,
    PaperShapingSection,
    GrainSynthesisSection,
    GuiConfigSection,
    HalationSection,
    ImageBannerWidget,
    InputImageSection,
    LoadRawSection,
    MetadataPanel,
    PreflashingSection,
    PreviewCropSection,
    PushPullSection,
    RollSection,
    ScanColorSection,
    ScanCurveSection,
    ScanFrameSection,
    ScanFinishingSection,
    ScanLevelsSection,
    ScanResponseSection,
    ScanToningSection,
    ScannerSection,
    SimpleDataclassSection,
    SimulationSection,
    SpecialSection,
    SpectralUpsamplingSection,
    TuneSection,
    _build_auxiliary_label,
    _build_button,
    _build_button_row,
    _build_collapsible_form_section,
    _build_linked_form_section,
    _build_vertical_container,
    _build_widget_label,
    _enum_values,
    _format_label,
    _new_form_layout,
    _set_single_collapsible_layout,
    _spec_row,
)


@dataclass(slots=True)
class WidgetBundle:
    filepicker: FilePickerSection
    # Alias of ``filepicker`` under the GuiState section name so the state
    # bridge (getattr by section name) reaches the Import RGB controls.
    load_rgb: FilePickerSection
    gui_config: GuiConfigSection
    config_defaults: 'ConfigDefaultsSection'
    locations: 'LocationsSection'
    display: DisplaySection
    input_image: InputImageSection
    load_raw: LoadRawSection
    grain: GrainSection
    grain_synthesis: 'GrainSynthesisSection'
    development: 'DevelopmentSection'
    paper_shaping: 'PaperShapingSection'
    bleach_bypass: 'BleachBypassSection'
    preflashing: PreflashingSection
    diffusion: DiffusionSection
    camera_diffusion: CameraDiffusionSection
    halation: HalationSection
    couplers: CouplersSection
    glare: GlareSection
    chemistry: ChemistrySection
    push_pull: PushPullSection
    aging: AgingSection
    scan_finishing: ScanFinishingSection
    scan_frame: ScanFrameSection
    scan_levels: ScanLevelsSection
    scan_response: ScanResponseSection
    scan_color: ScanColorSection
    scan_toning: ScanToningSection
    scan_curve: ScanCurveSection
    input_gamut_compress: InputGamutCompressSection
    output_gamut_compress: OutputGamutCompressSection
    special: SpecialSection
    paper: 'PaperTextureSection'
    simulation: SimulationSection
    preview_crop: PreviewCropSection
    camera: CameraSection
    exposure_control: ExposureControlSection
    enlarger: EnlargerSection
    scanner: ScannerSection
    spectral_upsampling: SpectralUpsamplingSection
    tune: TuneSection
    export: ExportSection
    roll: RollSection
    film_strip: FilmStripWidget
    gpu_backend: GpuBackendWidget
    banner: ImageBannerWidget
    metadata: MetadataPanel
    analysis: AnalysisPanel


def create_widget_bundle() -> WidgetBundle:
    bundle = _create_widget_bundle()
    # Phase 15: factory-default reset baselines on every section (profile
    # sync refreshes the synced fields' baselines per stock afterwards).
    for field_info in WidgetBundle.__dataclass_fields__.values():
        section = getattr(bundle, field_info.name)
        seed = getattr(section, 'seed_baselines', None)
        if callable(seed):
            seed()
    _wire_conditional_visibility(bundle)
    return bundle


def _wire_conditional_visibility(bundle: WidgetBundle) -> None:
    """NAT 2026-07-21: only show the settings that matter for the selected
    options — the Grain synthesis group only for the synthesis model, and
    the print-stage sections only while the print stage actually runs
    (scan-film mode bypasses it; positives auto-enable scan-film)."""
    def _sync_synthesis(model: str) -> None:
        bundle.grain_synthesis.setVisible(model == 'synthesis')

    bundle.grain.model.currentTextChanged.connect(_sync_synthesis)
    _sync_synthesis(bundle.grain.model.value)

    print_sections = (bundle.chemistry, bundle.paper_shaping, bundle.glare,
                      bundle.preflashing, bundle.diffusion, bundle.enlarger)

    def _sync_print(scan_film: bool) -> None:
        for section in print_sections:
            section.setVisible(not scan_film)

    bundle.simulation.bottom_scan_film.toggled.connect(_sync_print)
    _sync_print(bundle.simulation.bottom_scan_film.isChecked())

    # NAT 2026-07-22: the Development-time picker only means something for
    # B&W stocks (their profiles carry a development-time curve family) —
    # hidden for color stocks.
    def _sync_development(*_args):
        bundle.development.setVisible(
            _stock_is_bw(bundle.simulation.film_stock.value))

    bundle.simulation.film_stock.currentTextChanged.connect(_sync_development)
    _sync_development()


def _stock_is_bw(slug: object) -> bool:
    """Cheap channel_model probe: scan the profile JSON's head (the info
    block sits at the top; no full profile load), resolving custom slugs
    through their recipe's base stock. Unknown/broken -> color (visible
    surprises beat hidden controls)."""
    import json
    import re

    try:
        slug = str(slug)
        if slug.startswith('custom_'):
            from spektrafilm.profiles.custom_stocks import _recipe_path

            with _recipe_path(slug).open('r', encoding='utf-8') as file:
                slug = str(json.load(file).get('base_stock', ''))
        from importlib import resources as pkg_resources

        package = pkg_resources.files('spektrafilm.data.profiles')
        with (package / f'{slug}.json').open('r') as file:
            head = file.read(4096)
        match = re.search(r'"channel_model"\s*:\s*"(\w+)"', head)
        return bool(match) and match.group(1) == 'bw'
    except Exception:
        return False


def _create_widget_bundle() -> WidgetBundle:
    filepicker = FilePickerSection()
    input_image = InputImageSection(filepicker)
    scan_finishing = ScanFinishingSection()
    simulation = SimulationSection()
    # NAT 2026-07-22: spectral upsampling + taking filter render on MAIN
    # under Profiles (editors stay input_image-owned).
    simulation.adopt_spectral_rows(input_image)
    special = SpecialSection(simulation)
    glare = GlareSection()
    scanner = ScannerSection(simulation)
    simulation.bind_scan_for_print_glare_section(glare)
    grain = GrainSection()
    gui_config = GuiConfigSection()

    return WidgetBundle(
        filepicker=filepicker,
        load_rgb=filepicker,
        gui_config=gui_config,
        config_defaults=ConfigDefaultsSection(gui_config),
        locations=LocationsSection(),
        display=DisplaySection(),
        input_image=input_image,
        load_raw=LoadRawSection(),
        grain=grain,
        grain_synthesis=GrainSynthesisSection(grain),
        development=DevelopmentSection(simulation),
        paper_shaping=PaperShapingSection(simulation),
        bleach_bypass=BleachBypassSection(simulation),
        preflashing=PreflashingSection(),
        diffusion=DiffusionSection(simulation),
        camera_diffusion=CameraDiffusionSection(simulation),
        halation=HalationSection(),
        couplers=CouplersSection(),
        glare=glare,
        chemistry=ChemistrySection(),
        push_pull=PushPullSection(),
        aging=AgingSection(),
        scan_finishing=scan_finishing,
        scan_frame=ScanFrameSection(scan_finishing),
        scan_levels=ScanLevelsSection(scan_finishing),
        scan_response=ScanResponseSection(scan_finishing),
        scan_color=ScanColorSection(scan_finishing),
        scan_toning=ScanToningSection(scan_finishing),
        scan_curve=ScanCurveSection(scan_finishing),
        input_gamut_compress=InputGamutCompressSection(),
        output_gamut_compress=OutputGamutCompressSection(),
        special=special,
        paper=PaperTextureSection(),
        simulation=simulation,
        preview_crop=PreviewCropSection(input_image),
        camera=CameraSection(simulation),
        exposure_control=ExposureControlSection(simulation),
        enlarger=EnlargerSection(simulation),
        scanner=scanner,
        spectral_upsampling=SpectralUpsamplingSection(input_image),
        tune=TuneSection(special),
        export=ExportSection(simulation),
        roll=RollSection(),
        film_strip=FilmStripWidget(),
        gpu_backend=GpuBackendWidget(),
        banner=ImageBannerWidget(),
        metadata=MetadataPanel(),
        analysis=AnalysisPanel(),
    )