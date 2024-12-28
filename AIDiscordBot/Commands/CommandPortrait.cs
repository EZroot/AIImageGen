using Discord.WebSocket;
using DiscordMusicBot.Commands.Interfaces;
using Discord;
using DiscordMusicBot.Services.Interfaces;
using DiscordMusicBot.Services;

namespace DiscordMusicBot.Commands.Commands
{
    internal class CommandPortrait : IDiscordCommand
    {
        private string _commandName = "portrait";
        public string CommandName => _commandName;

        public SlashCommandBuilder Register()
        {
            return new SlashCommandBuilder()
            .WithName(_commandName)
            .AddOption("prompt", ApplicationCommandOptionType.String, "Prompt used to generate the image.", isRequired: true)
            .WithDescription("Generate a portrait with the given prompt");
        }

        public async Task ExecuteAsync(SocketSlashCommand command)
        {
            var prompt = command.Data.Options.First();
            await command.DeferAsync();
            var response = await Service.Get<IServiceRequestManager>().SendRequestAsync((string)prompt, 768, 1024);
            if(response[0] == '/' || response[0] == 'c' || response[0] == 'C')
                await command.FollowupWithFileAsync(response);
            else
                await command.FollowupAsync($"Error: {response}");
        }

    }
}
