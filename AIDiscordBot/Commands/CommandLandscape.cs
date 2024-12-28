using Discord.WebSocket;
using DiscordMusicBot.Commands.Interfaces;
using Discord;
using DiscordMusicBot.Services.Interfaces;
using DiscordMusicBot.Services;
using System.Diagnostics;

namespace DiscordMusicBot.Commands.Commands
{
    internal class CommandLandscape : IDiscordCommand
    {
        private string _commandName = "landscape";
        public string CommandName => _commandName;

        public SlashCommandBuilder Register()
        {
            return new SlashCommandBuilder()
            .WithName(_commandName)
            .AddOption("prompt", ApplicationCommandOptionType.String, "Prompt used to generate the image.", isRequired: true)
            .WithDescription("Generate a landscape with the given prompt");
        }

        public async Task ExecuteAsync(SocketSlashCommand command)
        {
            var prompt = command.Data.Options.First();
            await command.DeferAsync();
            var response = await Service.Get<IServiceRequestManager>().SendRequestAsync((string)prompt, 1024, 768, 50, "highly detailed, realistic, absurdres, highly-detailed, best quality, masterpiece, very aesthetic, landscape, wide-shot, ");
            if(response[0] == '/' || response[0] == 'c' || response[0] == 'C')
                await command.FollowupWithFileAsync(response);
            else
                await command.FollowupAsync($"Error: {response}");
        }
        
    }
}
