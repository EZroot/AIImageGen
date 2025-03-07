using DiscordMusicBot.Models;
using DiscordMusicBot.Services.Interfaces;
namespace DiscordMusicBot.Services
{
    internal class AnalyticsManager : IServiceAnalyticsManager
    {
        private AnalyticData _analyticData;
        public AnalyticData AnalyticData => _analyticData;

        public async Task Initialize()
        {
            _analyticData = Service.Get<IServiceDataManager>().LoadAnalytics();
            await Task.CompletedTask;
        }

        public async Task AddSongAnalytics(string userName, SongData songData)
        {
            var userAnalytics = _analyticData.UserAnalyticData.FirstOrDefault(u => u.UserName == userName);
            if (userAnalytics.UserName != null)
            {
                userAnalytics.SongHistory.Add(new SongAnlyticData { SongData = songData, NumberOfPlays = 1 });
                userAnalytics.SongHistory = userAnalytics.SongHistory.OrderBy(s => s.SongData.Title).ToList();
            }
            else
            {
                _analyticData.UserAnalyticData.Add(new UserAnalyticData
                {
                    UserName = userName,
                    SongHistory = new List<SongAnlyticData> { new SongAnlyticData { SongData = songData, NumberOfPlays = 1 } }
                });

                var newUserAnalytics = _analyticData.UserAnalyticData.First(u => u.UserName == userName);
                _analyticData.GlobalMostPlayedSongs = _analyticData.GlobalMostPlayedSongs.OrderByDescending(s => s.NumberOfPlays).ToList();
            }
            var globalSongData = _analyticData.GlobalMostPlayedSongs.FirstOrDefault(s => s.SongData.Title.Equals(songData.Title));
            if (globalSongData.SongData.Title == null || globalSongData.SongData.Title == "null")
            {
                _analyticData.GlobalMostPlayedSongs.Add(new SongAnlyticData { SongData = songData, NumberOfPlays = 1 });
            }
            else
            {
                globalSongData.NumberOfPlays++;
            }

            _analyticData.GlobalMostPlayedSongs = _analyticData.GlobalMostPlayedSongs.OrderBy(s => s.NumberOfPlays).ToList();

            for (int i = _analyticData.RecentSongHistory.Length - 1; i > 0; i--)
            {
                _analyticData.RecentSongHistory[i] = _analyticData.RecentSongHistory[i - 1]; 
            }
            _analyticData.RecentSongHistory[0] = songData;

            await Service.Get<IServiceDataManager>().SaveAnalytics(_analyticData);
        }
    }
}
