using DiscordMusicBot.Models;

namespace DiscordMusicBot.Services.Interfaces
{
    internal interface IServiceAnalyticsManager : IService
    {
        AnalyticData AnalyticData { get; }
        Task Initialize();
        Task AddSongAnalytics(string userName, SongData songData);
    }
}