namespace DiscordMusicBot.Services.Interfaces
{
    internal interface IServiceRequestManager : IService
    {
        Task Initialize();
        Task<string> SendRequestAsync(string prompt, int width = 768, int height = 1024, int inferenceSteps = 50,
                                                    string helperPrompt = "highly detailed, realistic, fine fabric detail, "
                                                                            + "absurdres, highly-detailed, best quality, masterpiece,"
                                                                            + "very aesthetic, portrait, ",
                                                    string negativePrompt = "lowres, worst quality, low quality, bad anatomy, bad hands,"
                                                                            + " multiple views, abstract, signature,"
                                                                            + " furry, anthro, bkub, 2koma, 4koma, comic, manga, sketch, ixy,");
    }
}